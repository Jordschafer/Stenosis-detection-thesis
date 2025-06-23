import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, detrend

# === CONFIGURATION ===
signal_dir = "logs/patient_signals"
patient_id = "14_024_3"  # Set this to a valid patient ID
apply_median_filter = True
filter_kernel_size = 5  # must be odd
save_path = f"logs/{patient_id}_pca_vs_tracking.png"

# === Load signals ===
pca_path = os.path.join(signal_dir, f"{patient_id}_pca.npy")
track_path = os.path.join(signal_dir, f"{patient_id}_tracking.npy")

if not os.path.exists(pca_path) or not os.path.exists(track_path):
    raise FileNotFoundError("PCA or tracking signal file not found.")

pca = np.load(pca_path)
tracking = np.load(track_path)

# === Trim to same length ===
N = min(len(pca), len(tracking))
pca = pca[:N]
tracking = tracking[:N]

# === Process PCA signal (same as visualize_pca_fft_patient) ===
pca = detrend(pca)
pca = (pca - np.mean(pca)) / np.std(pca)

# === Process tracking signal ===
tracking = detrend(tracking)
tracking = (tracking - np.mean(tracking)) / np.std(tracking)

if apply_median_filter:
    tracking = medfilt(tracking, kernel_size=filter_kernel_size)

# === Plot both signals ===
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(pca, lw=2, color='blue')
axs[0].set_title("PCA-gebaseerd ademhalingssignaal")
axs[0].set_ylabel("Genormaliseerde amplitude")
axs[0].grid(True)

axs[1].plot(tracking, lw=2, color='orange')
axs[1].set_title("Tracking-gebaseerd ademhalingssignaal (na SCA)")
axs[1].set_xlabel("Framenummer")
axs[1].set_ylabel("Genormaliseerde amplitude")
axs[1].grid(True)

plt.suptitle(f"Vergelijking ademhalingssignalen voor patiënt {patient_id}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(save_path, dpi=300)
plt.show()
print(f" Visualisatie opgeslagen als: {save_path}")
