import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend

# === Parameters ===
fps = 15
pca_path = "logs/pca_reference_signal.npy"
track_path = "logs/tracking_signal.npy"  # <- zie hieronder hoe dit ontstaat
output_plot = "logs/comparison_pca_vs_tracking.png"

# === Load signals ===
pca = np.load(pca_path)
tracking = np.load(track_path)

# === Align signals ===
N = min(len(pca), len(tracking))
pca = detrend(pca[:N])
tracking = detrend(tracking[:N])

# === Normalize for visual comparison ===
pca = (pca - np.mean(pca)) / np.std(pca)
tracking = (tracking - np.mean(tracking)) / np.std(tracking)

# === FFT ===
T = 1.0 / fps
freqs = rfftfreq(N, T)
fft_pca = np.abs(rfft(pca))
fft_track = np.abs(rfft(tracking))

# === Plot ===
plt.figure(figsize=(12, 5))

# Tijdsignaal
plt.subplot(1, 2, 1)
plt.plot(pca, label="PCA (intensiteit)", lw=2)
plt.plot(tracking, label="Tracking (bbox)", lw=2, alpha=0.75)
plt.title("Ademhalingssignaal (tijd)")
plt.xlabel("Frame")
plt.ylabel("Genormaliseerde amplitude")
plt.legend()
plt.grid(True)

# FFT
plt.subplot(1, 2, 2)
plt.plot(freqs, fft_pca, label="PCA FFT", lw=2)
plt.plot(freqs, fft_track, label="Tracking FFT", lw=2, linestyle="--")
plt.title("FFT van ademhalingssignalen")
plt.xlabel("Frequentie (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 1.5)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(output_plot)
plt.show()
print(f"Vergelijkingsplot opgeslagen naar: {output_plot}")
