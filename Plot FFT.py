import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend

# === Configuration ===
patient_id = "14_055_8"
fps = 15
pca_path = f"logs/patient_signals/{patient_id}_pca.npy"
track_path = f"logs/patient_signals/{patient_id}_tracking.npy"

# === Load and preprocess ===
pca = np.load(pca_path)
track = np.load(track_path)

N = min(len(pca), len(track))
pca = detrend(pca[:N])
track = detrend(track[:N])

# === Normalize ===
pca_norm = (pca - np.mean(pca)) / np.std(pca)
track_norm = (track - np.mean(track)) / np.std(track)

# === FFT ===
freqs = rfftfreq(N, 1.0 / fps)
pca_fft = np.abs(rfft(pca_norm))
track_fft = np.abs(rfft(track_norm))

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(freqs, pca_fft, label="PCA signal FFT", linewidth=2)
plt.plot(freqs, track_fft, label="Tracking signal FFT", linewidth=2)
plt.title(f"FFT Spectrum for Patient {patient_id}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
