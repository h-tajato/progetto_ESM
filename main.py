# main.py

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import latent_space_utils as lu

def main():
    device = "cuda"
    prompt_image = "a cat"
    prompt_audio = "a meowing cat"

    # Load models
    print("Loading pipelines...")
    sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to(device)

    # Text encodings
    emb_image = lu.encode_text(prompt_image, sd_pipe, device)
    au_pipe = StableDiffusionPipeline.from_pretrained("auffusion/auffusion-full-no-adapter", torch_dtype=torch.float32).to(device)
    
    emb_audio = lu.encode_text(prompt_audio, au_pipe, device)

    # Init and noise
    latents = lu.init_latents(sd_pipe, device=device)
    z_t, _ = lu.add_noise(sd_pipe, latents, steps=100)

    # Denoising with multimodal guidance
    print("Denoising...")
    z_denoised = lu.multimodal_denoise(z_t, emb_audio, emb_image, au_pipe, sd_pipe, steps=100)

    # Decode to spectrogram
    spectrogram = lu.decode_latents(z_denoised, sd_pipe)

    # Save image
    image_np = spectrogram.cpu().permute(0, 2, 3, 1).numpy()[0]
    image_pil = Image.fromarray((image_np * 255).astype("uint8"))
    image_pil.save("generated_spectrogram.png")

    # Convert to audio
    _ = lu.vocode_from_spectrogram(spectrogram)

if __name__ == "__main__":
    main()
