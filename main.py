from auffusion import AuffusionGuidance
import matplotlib.pyplot as plt
import torch
import numpy as np
import soundfile as sf
from spettrogram import *


if __name__ == '__main__':
    model = AuffusionGuidance(fp16=False)
    model = model.to(torch.device('cpu'))
    prompt_audio = 'coins falling on metal'
    prompt_video = 'a painting of a majestic mountain on the sea, grayscale'
    input_spectrogram = spectrify('.\\guida\\suono_monete.mp3')
    
    spect = model.prompt_to_spec(prompt_audio, prompt_video, height=256, width=1024, num_inference_steps=100, device=torch.device('cpu'), guidance_scale=10)

    print(torch.mean(spect), torch.min(spect), torch.max(spect))
    img = spect.detach().cpu().squeeze(0) 
    img = img.permute(1, 2, 0).numpy()
    # img = img.astype("16")
    img = img.clip(0, 1)
    imgs = (img * 255).round().astype('uint8')
    # print(img.shape, type(img))
    # gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]


    audio = model.spec_to_audio(spect.squeeze(0))
    sf.write('C:\\Users\\Flexo Rodriguez\\Desktop\\progetto_ESM\\prove\\ship.wav', np.ravel(audio), samplerate=16000)
    plt.figure()
    plt.imsave('C:\\Users\\Flexo Rodriguez\\Desktop\\progetto_ESM\\prove\\ship.png', img)
    # plt.show()
    # capito l'errore. Dobbiamo mettere fp16=False per usare doppia precisione, ma la GPU non regge, 

