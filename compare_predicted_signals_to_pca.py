import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from scipy.stats import pearsonr
from dtaidistance import dtw
import re

# === CONFIGURATION ===
models = ["frcnn_resnet101_frames", "frcnn_inceptionv3_frames"]
pca_dir = "logs/patient_signals"
track_dir_base = "logs/patient_signals_pred"
fps = 15
output_csv = "logs/pca_vs_predicted_metrics.csv"

def get_fft_peak(signal, fps):
    N = len(signal)
    freqs = rfftfreq(N, 1.0 / fps)
    fft_mag = np.abs(rfft(detrend(signal)))
    peak_freq = freqs[np.argmax(fft_mag)]
    snr = fft_mag.max() / (np.mean(fft_mag) + 1e-6)
    return peak_freq, snr

all_results = []

for model in models:
    track_dir = os.path.join(track_dir_base, model)
    print(f"\n🔍 Checking model: {model} → {track_dir}")

    # Only load per-patient files (no frame number in name)
    patient_ids = [
    f.replace("_tracking.npy", "")
    for f in os.listdir(track_dir)
    if f.endswith("_tracking.npy") and not re.search(r"_\d{4}\.bmp_tracking\.npy$", f)
]
    print(patient_ids)

    for pid in patient_ids:
        print(pid)
        try:
            pca_path = os.path.join(pca_dir, f"{pid}_pca.npy")
            track_path = os.path.join(track_dir, f"{pid}_tracking.npy")

            if not os.path.exists(pca_path) or not os.path.exists(track_path):
                print(f"[!] Skipping {pid}: missing PCA or tracking signal")
                continue

            pca = np.load(pca_path)
            track = np.load(track_path)

            N = min(len(pca), len(track))
            pca = detrend(pca[:N])
            track = detrend(track[:N])

            if not np.isfinite(track).all() or not np.isfinite(pca).all():
                print(f"[!] {model}/{pid} skipped: contains NaN or Inf")
                continue

            track_std = np.std(track)
            pca_std = np.std(pca)

            if track_std == 0 or pca_std == 0:
                print(f"[!] {model}/{pid} skipped: zero variance signal")
                continue

            track_norm = (track - np.mean(track)) / track_std
            pca_norm = (pca - np.mean(pca)) / pca_std

            pearson_corr, _ = pearsonr(pca_norm, track_norm)
            dtw_dist = dtw.distance(pca_norm, track_norm)
            pca_freq, pca_snr = get_fft_peak(pca_norm, fps)
            track_freq, track_snr = get_fft_peak(track_norm, fps)
            freq_diff = abs(pca_freq - track_freq)

            all_results.append({
                "model": model,
                "patient_id": pid,
                "frames": N,
                "pearson_r": round(pearson_corr, 3),
                "dtw_distance": round(dtw_dist, 3),
                "pca_freq_Hz": round(pca_freq, 3),
                "track_freq_Hz": round(track_freq, 3),
                "freq_diff_Hz": round(freq_diff, 3),
                "pca_snr": round(pca_snr, 2),
                "track_snr": round(track_snr, 2)
            })

        except Exception as e:
            print(f"[!] {model}/{pid} failed: {e}")

df = pd.DataFrame(all_results)
df.to_csv(output_csv, index=False)
print(f"\n✅ Results saved to {output_csv}")
