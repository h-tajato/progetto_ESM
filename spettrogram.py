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
    path = '.\\input_spectrograms\\prova.png'
    im = spectrify('.\\guida\\suono_monete.mp3')
    res = save_spectrogram(path, im)
    if (res):
        print('salvateggio eseguito con successo')
    else: 
        raise RuntimeError('Errore sconosciuto')

# # waveform, sample_rate = torchaudio.load("Suoni/coin-sound-79325.mp3")
# fs, segnale = None, None  # inizializzo per chiarezza

# fs = sample_rate
# # Se stereo, converto in mono prendendo la media dei canali
# segnale = waveform.mean(dim=0).numpy()

# # Normalizza
# if segnale.dtype == np.int16:
#     segnale = segnale / 32768

# # Parametri spettrogramma
# win_size = int(0.025 * fs)  # 25 ms
# hop_size = int(0.010 * fs)  # 10 ms
# num_frames = 1 + (len(segnale) - win_size) // hop_size

# # Finestra di Hamming per ridurre leakage
# finestra = np.hamming(win_size)

# # Costruiamo matrice spettrogramma (righe: frequenze, colonne: tempo)
# spettrogramma = []

# for i in range(num_frames):
#     start = i * hop_size
#     end = start + win_size
#     segmento = segnale[start:end]

#     if len(segmento) < win_size:
#         segmento = np.pad(segmento, (0, win_size - len(segmento)))

#     # Applica finestra e FFT
#     # segmento = np.mean(segmento, axis=1)
#     segment_windowed = segmento * finestra
#     fft_vals = np.fft.fft(segment_windowed)
#     fft_magn = np.abs(fft_vals[:win_size // 2])  # solo parte positiva

#     spettrogramma.append(fft_magn)

# # Converti in array (colonne: frame nel tempo, righe: frequenze)
# spettrogramma = np.array(spettrogramma).T  # shape: [frequenze, tempo]

# # Asse delle frequenze e del tempo
# frequenze = np.linspace(0, fs / 2, win_size // 2)
# tempi = np.arange(num_frames) * hop_size / fs

# # Crea asse temporale
# durata = len(segnale) / fs
# t = np.linspace(0, durata, len(segnale), endpoint=False)

# # Plot della sinusoide nel dominio del tempo
# plt.figure()
# plt.plot(t, segnale)
# plt.xlabel("Tempo [s]")
# plt.ylabel("Ampiezza [db]")
# plt.legend()

# # Visualizza
# plt.figure(figsize=(10, 4))
# plt.pcolormesh(tempi, frequenze, 20 * np.log10(spettrogramma + 1e-10), shading='gouraud', cmap='gray')
# plt.ylabel('Frequenza [Hz]')
# plt.xlabel('Tempo [s]')
# plt.title('Spettrogramma')
# plt.colorbar(label='Ampiezza [dB]')
# plt.tight_layout()
# plt.show()
