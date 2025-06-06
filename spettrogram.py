import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import torchaudio, torch

# Carica file WAV
# Carica file MP3

import torchaudio.transforms as T

def spectrify(audio_path: str):
    # Carica audio
    waveform, sample_rate = torchaudio.load(audio_path)  # [C, N]

    # Converti in mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, N]

    # Crop a 10 secondi se troppo lungo
    max_samples = 10 * sample_rate
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # Trasformazioni: MelSpectrogram + AmplitudeToDB
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    db_transform = T.AmplitudeToDB(top_db=80)

    # Applica trasformazioni
    mel_spec = mel_spectrogram(waveform)      # [1, n_mels, time]
    mel_spec_db = db_transform(mel_spec)      # [1, n_mels, time]

    return mel_spec_db.squeeze(0).numpy()  # tensore pronto per rete neurale


if __name__ == '__main__':
    im = spectrify('.\\guida\\suono_monete.mp3')

    plt.figure()
    plt.imshow(im, clim=None, cmap='jet')
    plt.title('Spettrogramma del Napule')
    plt.xlabel('[s]')
    plt.ylabel('[Mels]')
    plt.colorbar()
    plt.show()

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
