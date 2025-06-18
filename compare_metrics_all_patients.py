import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from scipy.stats import pearsonr
from dtaidistance import dtw


# === CONFIGURATION ===
signal_dir = "logs/patient_signals"
fps = 15
output_csv = "logs/respiratory_comparison_metrics_v2.csv"

def get_fft_peak(signal, fps):
    N = len(signal)
    freqs = rfftfreq(N, 1.0 / fps)
    fft_mag = np.abs(rfft(detrend(signal)))
    peak_freq = freqs[np.argmax(fft_mag)]
    snr = fft_mag.max() / (np.mean(fft_mag) + 1e-6)
    return peak_freq, snr

patients = [f.split("_pca.npy")[0] for f in os.listdir(signal_dir) if f.endswith("_pca.npy")]
results = []

for pid in sorted(set(patients)):
    try:
        pca = np.load(os.path.join(signal_dir, f"{pid}_pca.npy"))
        track = np.load(os.path.join(signal_dir, f"{pid}_tracking.npy"))
        N = min(len(pca), len(track))
        pca = detrend(pca[:N])
        track = detrend(track[:N])
        pca_norm = (pca - np.mean(pca)) / np.std(pca)
        track_norm = (track - np.mean(track)) / np.std(track)

        if np.std(pca) == 0 or np.std(track) == 0:
            print(f"[!] Skipped {pid}: constant signal")
            continue

        pearson_corr, _ = pearsonr(pca_norm, track_norm)
        dtw_dist = dtw.distance(pca_norm, track_norm)
        pca_freq, pca_snr = get_fft_peak(pca_norm, fps)
        track_freq, track_snr = get_fft_peak(track_norm, fps)
        freq_diff = abs(pca_freq - track_freq)

        results.append({
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
        print(f"[!] Failed on patient {pid}: {e}")

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"✅ Metrics written to: {output_csv}")
