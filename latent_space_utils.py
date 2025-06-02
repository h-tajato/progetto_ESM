# latent_utils.py

import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from converter import Generator, denormalize_spectrogram
from huggingface_hub import snapshot_download
from PIL import Image
import soundfile as sf
import os, gc

def encode_text(prompt, pipe, device):
    inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    return pipe.text_encoder(inputs.input_ids.to(device))[0]

def init_latents(pipe, shape=(1, 4, 64, 64), device="cuda"):
    return torch.randn(shape, device=device)

def add_noise(pipe, latents, steps=100):
    pipe.scheduler.set_timesteps(steps)
    t0 = pipe.scheduler.timesteps[0]
    noise = torch.randn_like(latents)
    return pipe.scheduler.add_noise(latents, noise=noise, timesteps=t0), t0

def classifier_free_guidance(unet, z_t, t, emb_cond, emb_uncond, guidance_scale):
    eps_uncond = unet(z_t, t, encoder_hidden_states=emb_uncond).sample
    eps_cond = unet(z_t, t, encoder_hidden_states=emb_cond).sample
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

def multimodal_denoise(z_t, emb_audio, emb_image, pipe_audio, pipe_image, steps=100, gamma_a=7.5, gamma_v=7.5):
    device = z_t.device
    pipe_audio.scheduler.set_timesteps(steps)
    pipe_image.scheduler.set_timesteps(steps)

    emb_uncond_audio = torch.zeros_like(emb_audio).to("cpu")
    emb_uncond_image = torch.zeros_like(emb_image).to("cpu")

    emb_audio = emb_audio.to("cpu")
    emb_image = emb_image.to("cpu")

    z_t = z_t.to("cpu")

    for t in pipe_audio.scheduler.timesteps:
        print(f"[Step {t}]")

        # ---- AUDIO GUIDANCE ----
        pipe_audio.unet.to("cuda")
        torch.cuda.empty_cache()
        z_t_gpu = z_t.to("cuda")
        eps_audio = classifier_free_guidance(
            pipe_audio.unet, z_t_gpu, t,
            emb_audio.to("cuda"),
            emb_uncond_audio.to("cuda"),
            gamma_a
        ).to("cpu")
        del pipe_audio.unet
        gc.collect()
        torch.cuda.empty_cache()

        # ---- IMAGE GUIDANCE ----
        pipe_image.unet.to("cuda")
        torch.cuda.empty_cache()
        z_t_gpu = z_t.to("cuda")
        eps_image = classifier_free_guidance(
            pipe_image.unet, z_t_gpu, t,
            emb_image.to("cuda"),
            emb_uncond_image.to("cuda"),
            gamma_v
        ).to("cpu")
        del pipe_image.unet
        gc.collect()
        torch.cuda.empty_cache()

        # ---- COMBINE AND STEP ----
        eps_combined = (eps_audio + eps_image) / 2.0
        pipe_image.scheduler.set_timesteps(steps)
        z_t = pipe_image.scheduler.step(eps_combined.to("cuda"), t, z_t.to("cuda")).prev_sample.to("cpu")

    return (1 / 0.18215) * z_t.to("cuda")

def decode_latents(latents_denoised, pipe):
    with torch.no_grad():
        image = pipe.vae.decode(latents_denoised).sample
    return image

def vocode_from_spectrogram(spectrogram, save_path="cat.wav"):
    device = 'cuda'
    image = (spectrogram / 2 + 0.5).clamp(0, 1)
    image_resized = torch.nn.functional.interpolate(image.mean(1, keepdim=True), size=(256, 1024), mode="bilinear")

    spec_tensor = image_resized.squeeze().cpu().unsqueeze(0)
    spec_tensor = denormalize_spectrogram(spec_tensor).float().to(device)

    model_path = "auffusion/auffusion-full-no-adapter"
    if not os.path.isdir(model_path):
        model_path = snapshot_download(model_path)

    vocoder = Generator.from_pretrained(model_path, subfolder="vocoder").to(device).float()

    print("[Generating Audio...]")
    audio = vocoder.inference(spec_tensor)

    sf.write(save_path, audio.squeeze(), samplerate=16000)
    return audio
