import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend

# === Configuration ===
patient_id = "14_055_8"
fps = 15
pca_path = f"logs/patient_signals/{patient_id}_pca.npy"
inception_path = f"logs/patient_signals_pred/frcnn_inceptionv3_frames/{patient_id}_tracking.npy"

# === Load and preprocess ===
pca = np.load(pca_path)
inception = np.load(inception_path)

N = min(len(pca), len(inception))
pca = detrend(pca[:N])
inception = detrend(inception[:N])

# === Normalize ===
pca_norm = (pca - np.mean(pca)) / np.std(pca)
inception_norm = (inception - np.mean(inception)) / np.std(inception)

# === FFT ===
freqs = rfftfreq(N, 1.0 / fps)
pca_fft = np.abs(rfft(pca_norm))
inception_fft = np.abs(rfft(inception_norm))

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(freqs, pca_fft, label="PCA signal FFT", linewidth=2)
plt.plot(freqs, inception_fft, label="InceptionV3 prediction FFT", linewidth=2)
plt.title(f"FFT Spectrum: PCA vs InceptionV3 ({patient_id})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
