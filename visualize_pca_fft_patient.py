import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend

# === CONFIGURATION ===
signal_dir = "logs/patient_signals"
patient_id = "14_076_3"  # <-- Change to a valid patient ID
fps = 15
save_path = f"logs/{patient_id}_pca_fft_visualization.png"

# === Load precomputed PCA signal ===
pca_path = os.path.join(signal_dir, f"{patient_id}_pca.npy")

if not os.path.exists(pca_path):
    raise FileNotFoundError(f"PCA signal file not found: {pca_path}")

pca_signal = np.load(pca_path)

# === Normalize and detrend ===
pca_signal = detrend(pca_signal)
pca_signal = (pca_signal - np.mean(pca_signal)) / np.std(pca_signal)

# === Compute FFT ===
N = len(pca_signal)
T = 1.0 / fps
freqs = rfftfreq(N, T)
fft_mag = np.abs(rfft(pca_signal))
peak_freq = freqs[np.argmax(fft_mag)]

# === Plot PCA signal and FFT ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Time-domain PCA signal
axs[0].plot(pca_signal, lw=2)
axs[0].set_title("PCA-signaal over tijd")
axs[0].set_xlabel("Frame")
axs[0].set_ylabel("Genormaliseerde amplitude")
axs[0].grid(True)

# Frequency-domain FFT
axs[1].plot(freqs, fft_mag, lw=2)
axs[1].axvline(peak_freq, color='red', linestyle='--', label=f"Piek: {peak_freq:.2f} Hz")
axs[1].set_xlim(0, 1.5)
axs[1].set_title("FFT van PCA-signaal")
axs[1].set_xlabel("Frequentie (Hz)")
axs[1].set_ylabel("Amplitude")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
print(f" Visualisatie opgeslagen als: {save_path}")
