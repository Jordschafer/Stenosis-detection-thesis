import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend

# === Configuration ===
signal_dir = "logs/patient_signals"
fps = 15
max_freq_hz = 1.5  # Optional limit for x-axis
pca_ffts = []
track_ffts = []

# === Loop over patients ===
for file in os.listdir(signal_dir):
    if file.endswith("_pca.npy"):
        patient_id = file.replace("_pca.npy", "")
        pca_path = os.path.join(signal_dir, f"{patient_id}_pca.npy")
        track_path = os.path.join(signal_dir, f"{patient_id}_tracking.npy")

        if not os.path.exists(track_path):
            continue

        pca = np.load(pca_path)
        track = np.load(track_path)

        N = min(len(pca), len(track))
        pca = detrend(pca[:N])
        track = detrend(track[:N])

        pca_norm = (pca - np.mean(pca)) / np.std(pca)
        track_norm = (track - np.mean(track)) / np.std(track)

        freqs = rfftfreq(N, 1.0 / fps)
        pca_fft = np.abs(rfft(pca_norm))
        track_fft = np.abs(rfft(track_norm))

        pca_ffts.append((freqs, pca_fft))
        track_ffts.append((freqs, track_fft))

# === Determine common x-range ===
min_len = min(len(f[1]) for f in pca_ffts + track_ffts)
freqs = pca_ffts[0][0][:min_len]

# === Create stacked FFT array ===
pca_matrix = np.array([f[1][:min_len] for f in pca_ffts])
track_matrix = np.array([f[1][:min_len] for f in track_ffts])

# === Plot ===
plt.figure(figsize=(10, 5))

for fft in pca_matrix:
    plt.plot(freqs, fft, color='blue', alpha=0.15)
for fft in track_matrix:
    plt.plot(freqs, fft, color='orange', alpha=0.15)

# Add mean FFT lines
plt.plot(freqs, pca_matrix.mean(axis=0), color='blue', label='Mean PCA FFT', linewidth=2)
plt.plot(freqs, track_matrix.mean(axis=0), color='orange', label='Mean Tracking FFT', linewidth=2)

plt.title("Overlay of FFT Spectra Across Patients (PCA vs Tracking)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, max_freq_hz)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
