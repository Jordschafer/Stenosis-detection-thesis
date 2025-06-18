import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIGURATION ===
metrics_csv = "logs/respiratory_comparison_metrics.csv"
output_dir = "logs/metric_plots"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
df = pd.read_csv(metrics_csv)

# === 1. Frequency difference boxplot ===
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["freq_diff_Hz"])
plt.title("Verschil in ademfrequentie (Hz) tussen PCA en tracking")
plt.ylabel("Frequentieverschil (Hz)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_freq_diff.png"))
plt.close()

# === 2. Pearson correlation histogram ===
plt.figure(figsize=(6, 4))
sns.histplot(df["pearson_r"], bins=15, kde=True)
plt.title("Correlatie tussen ademhalingssignalen (PCA vs tracking)")
plt.xlabel("Pearson correlatiecoëfficiënt")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hist_pearson_corr.png"))
plt.close()

# === 3. Scatter: PCA freq vs Tracking freq ===
plt.figure(figsize=(6, 6))
sns.scatterplot(x="pca_freq_Hz", y="track_freq_Hz", data=df)
plt.plot([0, 1], [0, 1], "r--", label="Ideale lijn")
plt.title("FFT piekfrequenties: PCA vs Tracking")
plt.xlabel("PCA frequentie (Hz)")
plt.ylabel("Tracking frequentie (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_pca_vs_tracking_freq.png"))
plt.close()

# === 4. Bland–Altman plot (FFT peak freq) ===
avg_freq = (df["pca_freq_Hz"] + df["track_freq_Hz"]) / 2
diff_freq = df["track_freq_Hz"] - df["pca_freq_Hz"]
mean_diff = diff_freq.mean()
std_diff = diff_freq.std()

plt.figure(figsize=(6, 4))
plt.scatter(avg_freq, diff_freq)
plt.axhline(mean_diff, color="red", linestyle="--", label="Gemiddeld verschil")
plt.axhline(mean_diff + 1.96 * std_diff, color="gray", linestyle=":")
plt.axhline(mean_diff - 1.96 * std_diff, color="gray", linestyle=":")
plt.title("Bland–Altman plot: Tracking - PCA FFT frequentie")
plt.xlabel("Gemiddelde frequentie (Hz)")
plt.ylabel("Verschil (Tracking - PCA) in Hz")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bland_altman_freq.png"))
plt.close()

print(f"✅ Plots opgeslagen in: {output_dir}")
