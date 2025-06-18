import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV to inspect the structure
df = pd.read_csv("logs/pca_vs_predicted_metrics.csv")
print(df.head())

df_resnet = df[df["model"] == "frcnn_resnet101_frames"].copy()
df_inception = df[df["model"] == "frcnn_inceptionv3_frames"].copy()

# Compute mean and std for all columns except the first
numeric_part = df_inception.iloc[:, 2:]  # exclude first column
mean_row = numeric_part.mean()
std_row = numeric_part.std()

# Add a label for the first column
mean_row[df_inception.columns[0]] = "mean"
std_row[df_inception.columns[0]] = "std"

# Convert to full rows with same columns
df_inception = pd.concat([df_inception, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)

# Print or save
print(df_inception.head())
print(df_inception.tail())

# Select key columns for boxplot
metrics_df = df_inception.rename(columns={
    'pearson_r': 'Pearson Correlation',
    'dtw_distance': 'DTW Distance',
    'freq_diff_Hz': 'Frequency Error (Hz)',
    'pca_snr': 'PCA SNR (dB)',
    'track_snr': 'GT Tracking SNR (dB)'
})

# Melt for seaborn boxplot
melted_df = metrics_df.melt(
    id_vars='patient_id',
    value_vars=[
        'Pearson Correlation',
        'DTW Distance',
        'Frequency Error (Hz)',
        'PCA SNR (dB)',
        'GT Tracking SNR (dB)',
    ],
    var_name='Metric',
    value_name='Value'
)

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=melted_df, x='Metric', y='Value', palette='pastel')
plt.title('Distribution of PCA vs Faster R-CNN + ResNet101 Comparison Metrics')
plt.ylabel('Metric Value')
plt.xticks(rotation=15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()