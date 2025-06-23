import os
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

# === Pad naar testset ===
image_dir = "split_data/test/images"
output_path = "logs/pca_reference_signal.npy"

# === Laad en sorteer alle BMP frames ===
frame_paths = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".bmp")])
if not frame_paths:
    raise RuntimeError("Geen .bmp bestanden gevonden in test/images")

# === Open eerste frame om afmetingen te bepalen ===
example = Image.open(os.path.join(image_dir, frame_paths[0])).convert("L")
height, width = example.size
num_pixels = example.size[0] * example.size[1]

# === Stapel beelden in een matrix: shape = (n_frames, n_pixels) ===
frames = []
for fname in frame_paths:
    img = Image.open(os.path.join(image_dir, fname)).convert("L")
    img_resized = img.resize((300, 300))  # kies gewenste vaste resolutie
    img_np = np.array(img_resized, dtype=np.float32).flatten() / 255.0
    frames.append(img_np)

X = np.stack(frames, axis=0)  # shape: (n_frames, n_pixels)

# === PCA toepassen (over tijd) ===
pca = PCA(n_components=1)
principal_signal = pca.fit_transform(X)  # shape: (n_frames, 1)
principal_signal = principal_signal.flatten()

# === Opslaan en plotten ===
os.makedirs("logs", exist_ok=True)
np.save(output_path, principal_signal)

plt.plot(principal_signal)
plt.title("PCA-basis ademhalingssignaal")
plt.xlabel("Frame")
plt.ylabel("PCA-component 1")
plt.tight_layout()
plt.savefig("logs/pca_reference_plot.png")
plt.show()

print(f"PCA-signaal opgeslagen in {output_path}")
