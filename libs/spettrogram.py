import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import torchaudio, torch

# Carica file WAV
# Carica file MP3

import torchaudio.transforms as T

import torch
import torchaudio
import torchaudio.transforms as T

def spectrify(audio_path: str):
    waveform, sample_rate = torchaudio.load(audio_path)  # [C, N]

    # Converti in mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Calcola lunghezza target per ottenere 1024 frame
    n_fft = 1024
    hop_length = 512
    target_frames = 1024
    target_samples = hop_length * (target_frames - 1) + n_fft  # = 524800

    # Crop o padding per avere esattamente target_samples
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    elif waveform.shape[1] < target_samples:
        pad_amount = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    # Crea le trasformazioni
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=256  # altezza desiderata
    )
    db_transform = T.AmplitudeToDB(top_db=80)

    # Applica le trasformazioni
    mel_spec = mel_spectrogram(waveform)      # [1, 256, 1024]
    mel_spec_db = db_transform(mel_spec)      # [1, 256, 1024]

    return mel_spec_db.squeeze(0).numpy(), mel_spec_db.squeeze(0).shape



def save_spectrogram(path:str, spect, clip=False)->bool:
    if(clip):
        h, w = spect.shape[0], spect.shape[1]
        
    
    plt.figure()
    plt.imshow(spect, cmap='gray')
    plt.axis('off')

    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    return True

if __name__ == '__main__':
    path = '.\\input_spectrograms\\elephant\\elephant.png'
    im, shape= spectrify("C:\\Users\\Flexo Rodriguez\\Desktop\\progetto_ESM\\input_spectrograms\\elephant\\elephant.mp3")
    res = save_spectrogram(path, im)
    if (res):
        print(f'salvateggio eseguito con successo con shape {shape}')
    else: 
        raise RuntimeError('Errore sconosciuto')