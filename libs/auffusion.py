# Librerie e funzioni

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils

from torchvision.utils import save_image
from pathlib import Path
from libs.converter import normalize_img
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from lightning import seed_everything
from libs.auffusion_converter import Generator, denormalize_spectrogram

# Identificazione della cartella di main effettiva tramite la ricerca del file .project-root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Funzione di heavside per cui: 
# H(x) = 1 (x >= 0)
# H(x) = 0 (x < 0)
def heaviside(x):
        if x>=0:
            return 1
        else:
            return 0

# Dichiarazione della classe utilizzata
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

    # Funzione di generazione dei latenti tramire iterazioni di denoising con la U-NET
    @torch.no_grad()
    def produce_latents(self, text_embeddings_au, text_embeddings_sd, 
                        height=512, width=512, num_inference_steps=50, 
                        guidance_scale_audio=7.5,guidance_scale_video=7.5, 
                        latents=None, generator=None, strength=0.8):

        # Definizione della variabile per identificare lo "stato di partenza" del denoising
        T = 991

        # Gestione del caso guidato e non guidato
        if latents is None:
            # Parametri per comprendere l'incidenza, uno solo tra i due valori (indipendenza da chi), può assumere valori nell'intervallo [0,1]
            t_a, t_v = 1.0, 0.9
            
            # Generazione, nel caso non guidato, del rumore gaussiano a media 0 e a varianza 1 (o identica)
            latents = torch.randn((text_embeddings_au.shape[0] // 2, self.unet.config.in_channels, height // 8, width // 8), generator=generator, dtype=self.unet.dtype).to(text_embeddings_au.device)
        else:
            t_a, t_v = 0.5, 1.0
            
            # Generazione del rumore, aggiungerndo al riferimento dato in ingresso un rumore ben definito dallo stato della prima iterazione
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), 
                                                torch.tensor([T], dtype=torch.long).to(text_embeddings_au.device))
            if torch.isnan(latents).any():
                raise ValueError("Il tensore contiene NaN!")

        # Setting del timestep, nel caso strength sia impostato, allora si avrà do conseguenza una "rifuzione" di iterazioni di denoising (permette di decidere l'incidenza del rumore di riferimento inserito)
        self.scheduler.set_timesteps(num_inference_steps)
        t_start = int(num_inference_steps*(1-strength))
        timesteps = self.scheduler.timesteps[t_start:]

        # Ciclo di denoising
        for i, t in enumerate(timesteps):
            # Print di servizio(per controllare le iterzioni svolte)
            print(f'[SCHEDULER]\t-\titerazione di denoising {t.item()},\ti={i}')
            
            # Concatenazione dei vettori latenti
            latent_model_input = torch.cat([latents] * 2)

            # APPLICAZIONE DEL MODELLO DI DENOISING
            
            # -----> Processing audio <----- #
            # Predizione del rumore tramite la unet
            noise_pred_au = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_au)['sample']

            # Ricavo il rumore predetto condizionato dal prompt e non condizionato dal prompt 
            noise_pred_uncond_au, noise_pred_cond_au = noise_pred_au.chunk(2)

            # Vado a processare il rumore in base a quello che ho ottenuto (rumore condizionato e non condizionato)
            # prodotto di classifier free-guidance
            noise_pred_au = noise_pred_uncond_au + guidance_scale_audio * (noise_pred_cond_au - noise_pred_uncond_au)

            # -----> Processing video <----- #
            # Predizione del rumore tramite la unet
            noise_pred_sd = self.unet_sd(latent_model_input, t, encoder_hidden_states=text_embeddings_sd)['sample']

            # Ricavo il rumore condizionato dal prompt e non condizionato dal prompt 
            noise_pred_uncond_sd, noise_pred_cond_sd = noise_pred_sd.chunk(2)

            # Vado a processare il rumore in base a quello che ho ottenuto (rumore condizionato e non condizionato)
            # prodotto di classifier free-guidance
            noise_pred_sd = noise_pred_uncond_sd + guidance_scale_video * (noise_pred_cond_sd - noise_pred_uncond_sd)  

            # -----> Implementazione del paper <----- #

            # Calcolo dei coefficenti omega_v ed omega_a di cui parla il paper per il calcolo dei lambda
            omega_a = heaviside(t_a*T-t.item()) 
            omega_v = heaviside(t_v*T-t.item())

            # Calcolo dei lambda in base ai precedenti parametri
            lambda_a = omega_a / (omega_a+omega_v) 
            lambda_v = omega_v / (omega_a+omega_v) 

            # Media "pesata" dei rumori ottenuti dalle due unet
            noise_pred = lambda_a*noise_pred_au + lambda_v*noise_pred_sd

            # Controllo per eventuali valori NaN (può capitare se eseguito su GPU più scarse in prestazioni)
            if torch.isnan(noise_pred).any():
                raise ValueError("Il tensore rumore contiene NaN!")

            # Generazione del nuovo latents da processare tramite la "rimozione" del rumore con lo schedulers
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):
        # Vado ad adattare il latens alla tipologia di dato che può processare il VAE
        latents = latents.to(self.vae.dtype)

        # Vado a riscalare il valore del latents (compenso dello scalaggio in fase di encoding per evitare valori troppo piccoli)
        latents = 1 / self.vae.config.scaling_factor * latents

        # Decodifica effettiva del dato
        imgs = self.vae.decode(latents).sample

        # Dato che ho valori compresi tra -1 ed 1, vado a restringere il range tra 0 ed 1
        # precisamente, la prima divisione riporta i valori nel range [-0.5, 0.5],
        # mentre +0.5 riporta tale range a [0,1] per poi in fine, per sicurezza, utilizzare
        # la funzione clamp, che taglia tutti i valori fuori range riportandoli a [0, 1]
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        
        # Riporto l'immagine dall'essere tra 0 ed 1 (considerando un immagine normalizzata) a -1 ed 1
        imgs = 2 * imgs - 1

        # Vado a calcolare il vettore latenti considerato
        posterior = self.vae.encode(imgs).latent_dist
        
        # Vado a scalare tale valore per evitare che i valori ottenuti, se troppo piccoli, vengano ignorati (zero crossing)
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def spec_to_audio(self, spec):

        # Carico lo spettrogramma sulla GPU
        spec = spec.to(dtype=self.precision_t)

        # Vado a denoramlizzare lo spettrogramma
        denorm_spec = denormalize_spectrogram(spec)

        # Calcolo dell'audio tramite il vocoder importato
        audio = self.vocoder.inference(denorm_spec)

        return audio

    def prompt_to_spec(self, prompt_au, prompt_sd, 
                       negative_prompt_au='',negative_prompt_sd = '',
                       height=512, width=512, 
                       num_inference_steps=50, guidance_scale_audio=7.5, guidance_scale_video=7.5, 
                       latents=None, device=None, generator=None, input_spectrogram=None, strength=0.8):
        
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
                                        guidance_scale_video=guidance_scale_video, generator=generator, strength=0.0) # [1, 4, 64, 64]
            except ValueError as e:
                raise e
            # Img latents -> imgs
        
        else:
            # input spectrogram è un immagine (256,1024). Convertiamola a tensore 
            # rgb_spect = gray2rgb(input_spectrogram.astype(np.uint8))  # assicura [0,255]

            rgb_spect = normalize_img(input_spectrogram)  # shape (H, W, 3), dtype float32
            spect_tensor = torch.from_numpy(rgb_spect).permute(2, 0, 1).unsqueeze(0)  # shape (1, 3, H, W)
            spect_tensor = spect_tensor.to(dtype=self.unet.dtype, device=self.unet.device)

            # rgb_spect = np.float16(normalize_img(input_spectrogram)) / np.max(input_spectrogram)
            # spect_tensor = torch.from_numpy(rgb_spect).permute(2,0,1).to(dtype=self.unet.dtype) 
            # spect_tensor = spect_tensor.squeeze(0).to(self.unet.device)

            # Encode (assumendo sia un VAE)
            latent_1 = self.encode_imgs(spect_tensor)
            latent_1 = latent_1.to(self.unet.device, dtype=self.unet.dtype)
            torch.cuda.empty_cache()
            gc.collect()
            try:
                latents = self.produce_latents(text_embeds_au, text_embeds_sd, height=height, 
                                        width=width, num_inference_steps=num_inference_steps, 
                                        guidance_scale_audio=guidance_scale_audio, 
                                        guidance_scale_video=guidance_scale_video, generator=generator, latents=latent_1,strength=strength)
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