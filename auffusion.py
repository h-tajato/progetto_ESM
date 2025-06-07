from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import gc

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

# from .perpneg_utils import weighted_perpendicular_aggregator
import numpy as np
from lightning import seed_everything

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from auffusion_converter import Generator, denormalize_spectrogram

def heaviside(x):
        if x>=0:
            return 1
        else:
            return 0

class AuffusionGuidance(nn.Module):
    def __init__(
        self, 
        repo_id_auffusion='auffusion/auffusion-full-no-adapter',
        repo_id_stable = 'runwayml/stable-diffusion-v1-5',
        fp16=True,
        t_range=[0.02, 0.98],
        **kwargs
    ):
        super().__init__()

        self.repo_id = repo_id_auffusion

        self.precision_t = torch.float16 if fp16 else torch.float32

        ## Creazione modello totale

        # import modello auffusion
        self.vae, self.tokenizer, self.text_encoder, self.unet = self.create_model_from_pipe(repo_id_auffusion, self.precision_t)
        self.scheduler = DDIMScheduler.from_pretrained(repo_id_auffusion, subfolder="scheduler", torch_dtype=self.precision_t)
        self.vocoder = Generator.from_pretrained(repo_id_auffusion, subfolder="vocoder").to(dtype=self.precision_t)

        self.register_buffer('alphas_cumprod', self.scheduler.alphas_cumprod)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        # import modello stable diffusion
        self.unet_sd = UNet2DConditionModel.from_pretrained(repo_id_stable, subfolder='unet', torch_dtype=self.precision_t)

    def create_model_from_pipe(self, repo_id, dtype):
        # Carica la pipe (i pesi) tramite hugging face
        pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=dtype)

        # Estrae il vae
        vae = pipe.vae

        # Estrae il tokenizer
        tokenizer = pipe.tokenizer

        # Estrae l'encoder
        text_encoder = pipe.text_encoder

        # Estrae la unet
        unet = pipe.unet

        return vae, tokenizer, text_encoder, unet

    @torch.no_grad()
    def get_text_embeds(self, prompt, device):
        # prompt: [str]
        # import pdb; pdb.set_trace()

        # Vado a tokenizzare i valori
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')


        prompt_embeds = self.text_encoder(inputs.input_ids.to(device))[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        return prompt_embeds

    

    def train_step(self, text_embeddings, pred_spec, guidance_scale=100, as_latent=False, t=None, grad_scale=1, save_guidance_path:Path=None):
        # import pdb; pdb.set_trace()
        pred_spec = pred_spec.to(self.vae.dtype)

        if as_latent:
            latents = pred_spec
        else:    
            if pred_spec.shape[1] != 3:
                pred_spec = pred_spec.repeat(1, 3, 1, 1)

            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_spec)

        if t is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=latents.device)
        else:
            t = t.to(dtype=torch.long, device=latents.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas_cumprod[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents.device)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(latents.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(latents.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss


    @torch.no_grad()
    def produce_latents(self, text_embeddings_au, text_embeddings_sd, 
                        height=512, width=512, num_inference_steps=50, 
                        guidance_scale_audio=7.5,guidance_scale_video=7.5, 
                        latents=None, generator=None):

        # definizione costanti utili per la media pesata iterativa
        T = 991
        if latents is None:
            t_a, t_v = 1.0, 0.9
            latents = torch.randn((text_embeddings_au.shape[0] // 2, self.unet.config.in_channels, height // 8, width // 8), generator=generator, dtype=self.unet.dtype).to(text_embeddings_au.device)
        else:
            t_a, t_v = 0.5, 1.0
            noise_par = 0.9
            # Genera rumore come nel ramo if
            noise = torch.randn(latents.shape, generator=generator, dtype=self.unet.dtype).to(latents.device)

            # print("Rumore aggiunto al latente in input (manuale, senza scheduler)")
            latents = latents + noise_par * noise  # 0.1 è il livello di rumore, regola a piacere

            if torch.isnan(latents).any():
                raise ValueError("Il tensore contiene NaN!")
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # print(f'[SCHEDULER]\t-\tt:  {t.item()}\tt shape: {t.shape}')
            print(f'[SCHEDULER]\t-\titerazione di denoising {t.item()},\ti={i}')
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # fase auffusion

            # predict the noise residual
            noise_pred_au = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_au)['sample']

            # perform guidance
            noise_pred_uncond_au, noise_pred_cond_au = noise_pred_au.chunk(2)
            noise_pred_au = noise_pred_uncond_au + guidance_scale_audio * (noise_pred_cond_au - noise_pred_uncond_au)

            # fase stable diffusion

            # predict the noise residual
            noise_pred_sd = self.unet_sd(latent_model_input, t, encoder_hidden_states=text_embeddings_sd)['sample']

            # perform guidance
            noise_pred_uncond_sd, noise_pred_cond_sd = noise_pred_sd.chunk(2)
            noise_pred_sd = noise_pred_uncond_sd + guidance_scale_video * (noise_pred_cond_sd - noise_pred_uncond_sd)  

            # media 
            omega_a = heaviside(t_a*T-t.item()) 
            omega_v = heaviside(t_v*T-t.item())

            lambda_a = omega_a / (omega_a+omega_v) 
            lambda_v = omega_v / (omega_a+omega_v) 

            noise_pred = lambda_a*noise_pred_au + lambda_v*noise_pred_sd

            if torch.isnan(noise_pred).any():
                raise ValueError("Il tensore rumore contiene NaN!")

            #  print(f'[SCHEDULER]\t-\t(T, t, l_a, l_b) = ({T, t.item(), lambda_a, lambda_v})')

            # print(f'[SCHEDULER]\tnoise pred a: {noise_pred_cond_au, noise_pred_uncond_au}')
            # print(f'[SCHEDULER]\tnoise pred v: {noise_pred_cond_sd, noise_pred_uncond_sd}')
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):
        # Impostazione del latente ricevuto in ingresso secondo il dtype dell'encoder
        latents = latents.to(self.vae.dtype)

        # vado a rescalare il latent del decoder secondo il fattore impostato
        latents = 1 / self.vae.config.scaling_factor * latents

        # Vado a decodificare il latent tramite l'encoder
        imgs = self.vae.decode(latents).sample

        # Dato che ho valori compresi tra -1 ed 1, vado a restringere il range tra 0 ed 1
        # precisamente, la prima divisione riporta i valori nel range [-0.5, 0.5],
        # mentre +0.5 riporta tale range a [0,1] per poi in fine, per sicurezza, utilizzare
        # la funzione clamp, che taglia tutti i valori fuori range riportandoli a [0, 1]
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        # Operazione inversa, per portare il range da [0,1] a [-1,1]
        imgs = 2 * imgs - 1

        # Calcolo del vettore latente tramite l'encoder
        posterior = self.vae.encode(imgs).latent_dist
        
        # Moltiplicazione per il fattore di scala
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def spec_to_audio(self, spec):
        spec = spec.to(dtype=self.precision_t)
        denorm_spec = denormalize_spectrogram(spec)
        audio = self.vocoder.inference(denorm_spec)
        return audio

    def prompt_to_audio(self, prompt_au, prompt_sd, 
                       negative_prompt_au='',negative_prompt_sd = '',
                       height=512, width=512, 
                       num_inference_steps=50, guidance_scale=7.5, 
                       latents=None, device=None, generator=None):
        
        if isinstance(prompt_au, str) and isinstance(prompt_sd, str):
            prompt_au = [prompt_au]
            prompt_sd = [prompt_sd]

        if isinstance(negative_prompt_au, str) and isinstance(negative_prompt_sd, str):
            negative_prompt_au = [negative_prompt_au]
            negative_prompt_sd = [negative_prompt_sd]

        # Prompt auffusion -> text embeds auffusion
        pos_embeds_au = self.get_text_embeds(prompt_au, device) # [1, 77, 768]
        neg_embeds_au = self.get_text_embeds(negative_prompt_au, device)
        text_embeds_au = torch.cat([neg_embeds_au, pos_embeds_au], dim=0) # [2, 77, 768]

        # Prompt stable diffusion -> text embeds stable diffusion
        pos_embeds_sd = self.get_text_embeds(prompt_sd, device) # [1, 77, 768]
        neg_embeds_sd = self.get_text_embeds(negative_prompt_sd, device)
        text_embeds_sd = torch.cat([neg_embeds_sd, pos_embeds_sd], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds_au, text_embeds_sd, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        spec = imgs[0]
        denorm_spec = denormalize_spectrogram(spec)
        audio = self.vocoder.inference(denorm_spec)
        # # Img to Numpy
        # imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).round().astype('uint8')

        return audio

    def prompt_to_spec(self, prompt_au, prompt_sd, 
                       negative_prompt_au='',negative_prompt_sd = '',
                       height=512, width=512, 
                       num_inference_steps=50, guidance_scale_audio=7.5, guidance_scale_video=7.5, 
                       latents=None, device=None, generator=None, input_spectrogram=None):
        
        if isinstance(prompt_au, str) and isinstance(prompt_sd, str):
            prompt_au = [prompt_au]
            prompt_sd = [prompt_sd]

        if isinstance(negative_prompt_au, str) and isinstance(negative_prompt_sd, str):
            negative_prompt_au = [negative_prompt_au]
            negative_prompt_sd = [negative_prompt_sd]

            # Prompt auffusion -> text embeds auffusion
            pos_embeds_au = self.get_text_embeds(prompt_au, device) # [1, 77, 768]
            neg_embeds_au = self.get_text_embeds(negative_prompt_au, device)
            text_embeds_au = torch.cat([neg_embeds_au, pos_embeds_au], dim=0) # [2, 77, 768]

            # Prompt stable diffusion -> text embeds stable diffusion
            pos_embeds_sd = self.get_text_embeds(prompt_sd, device) # [1, 77, 768]
            neg_embeds_sd = self.get_text_embeds(negative_prompt_sd, device)
            text_embeds_sd = torch.cat([neg_embeds_sd, pos_embeds_sd], dim=0) # [2, 77, 768]

        if input_spectrogram is None:
            # Text embeds -> img latents
            try:
                latents = self.produce_latents(text_embeds_au, text_embeds_sd, height=height, 
                                        width=width, latents=latents, num_inference_steps=num_inference_steps, 
                                        guidance_scale_audio=guidance_scale_audio, 
                                        guidance_scale_video=guidance_scale_video, generator=generator) # [1, 4, 64, 64]
            except ValueError as e:
                raise e
            # Img latents -> imgs
        
        else:
            # input spectrogram è un immagine (256,1024). Convertiamola a tensore 
            gray_spect = input_spectrogram[..., np.newaxis]
            rgb_spect = np.repeat(gray_spect, 3, axis=2)

            spect_tensor = torch.from_numpy(rgb_spect).to(dtype=self.precision_t) / 255.0
            spect_tensor = spect_tensor.permute(2, 0, 1).unsqueeze(0).to(torch.device('cuda'))

            latent_1 = self.encode_imgs(spect_tensor)

            torch.cuda.empty_cache()
            gc.collect()
            try:
                latents = self.produce_latents(text_embeds_au, text_embeds_sd, height=height, 
                                        width=width, num_inference_steps=num_inference_steps, 
                                        guidance_scale_audio=guidance_scale_audio, 
                                        guidance_scale_video=guidance_scale_video, generator=generator, latents=latent_1)
            except ValueError as e:
                raise RuntimeError('Errore nella generazione del tensore latente')
            

        torch.cuda.empty_cache()
        gc.collect()
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs
    
    # def prompt_image_to_spect(self, prompt_au, prompt_sd, 
    #                    negative_prompt_au='',negative_prompt_sd = '',
    #                    height=512, width=512, 
    #                    num_inference_steps=50, guidance_scale=7.5, 
    #                    latents=None, device=None, generator=None, input_spectrogram=None):
        
    #     if isinstance(prompt_au, str) and isinstance(prompt_sd, str):
    #         prompt_au = [prompt_au]
    #         prompt_sd = [prompt_sd]

    #     if isinstance(negative_prompt_au, str) and isinstance(negative_prompt_sd, str):
    #         negative_prompt_au = [negative_prompt_au]
    #         negative_prompt_sd = [negative_prompt_sd]

    #     if input_spectrogram is None:
    #         raise ValueError('Errore! chiama prompt to to spec invece di questa')
        
    #     spect_tensor = torch.from_numpy(input_spectrogram).float() / 255.0

    #     spect_tensor = spect_tensor.permute(2, 0, 1).unsqueeze(0)
    #     latent_1 = self.encode_imgs(spect_tensor)

    #     torch.cuda.empty_cache()
    #     gc.collect()
        

    #     # Prompt auffusion -> text embeds auffusion
    #     pos_embeds_au = self.get_text_embeds(prompt_au, device) # [1, 77, 768]
    #     neg_embeds_au = self.get_text_embeds(negative_prompt_au, device)
    #     text_embeds_au = torch.cat([neg_embeds_au, pos_embeds_au], dim=0) # [2, 77, 768]

    #     # Prompt stable diffusion -> text embeds stable diffusion
    #     pos_embeds_sd = self.get_text_embeds(prompt_sd, device) # [1, 77, 768]
    #     neg_embeds_sd = self.get_text_embeds(negative_prompt_sd, device)
    #     text_embeds_sd = torch.cat([neg_embeds_sd, pos_embeds_sd], dim=0) # [2, 77, 768]

    #     # Text embeds -> img latents
    #     latents = self.produce_latents(text_embeds_au, text_embeds_sd, height=height, width=width, latents=latent_1, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator) # [1, 4, 64, 64]

    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     # Img latents -> imgs
    #     imgs = self.decode_latents(latents) # [1, 3, 512, 512]
    #     return imgs

if __name__ == '__main__':
    import numpy as np
    import argparse
    from PIL import Image
    import os
    import soundfile as sf

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='A kitten mewing for attention')
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--repo_id', type=str, default='auffusion/auffusion-full-no-adapter', help="stable diffusion version")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--out_dir', type=str, default='logs/test')

    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = AuffusionGuidance(repo_id=opt.repo_id, fp16=opt.fp16)
    sd = sd.to(device)

    audio = sd.prompt_to_audio(opt.prompt, opt.negative, opt.H, opt.W, opt.steps, device=device)
    # import pdb; pdb.set_trace()
    # visualize audio
    save_folder = opt.out_dir
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f'{opt.prompt}.wav')
    sf.write(save_path, np.ravel(audio), samplerate=16000)