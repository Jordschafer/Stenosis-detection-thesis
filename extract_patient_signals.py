import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET

# === CONFIGURATION ===
image_dir = "split_data/test/images"
annotation_dir = "split_data/test/annotations"
output_dir = "logs/patient_signals"
resize_shape = (300, 300)  # all images resized to this for PCA
os.makedirs(output_dir, exist_ok=True)

def get_patient_id(filename):
    parts = filename.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else "unknown"

def parse_centroid(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
        obj = root.find("object")
        if obj is None:
            return None
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        return (cx, cy)
    except:
        return None

# === Step 1: Group frames by patient ===
patient_frames = {}
for fname in sorted(os.listdir(image_dir)):
    if not fname.lower().endswith(".bmp"):
        continue
    pid = get_patient_id(fname)
    patient_frames.setdefault(pid, []).append(fname)

# === Step 2: Process each patient ===
for pid, frames in patient_frames.items():
    print(f"Processing patient {pid} ({len(frames)} frames)")
    X = []
    centroids = []

    for fname in frames:
        img_path = os.path.join(image_dir, fname)
        xml_path = os.path.join(annotation_dir, os.path.splitext(fname)[0] + ".XML")

        # PCA input image
        img = Image.open(img_path).convert("L").resize(resize_shape)
        X.append(np.array(img, dtype=np.float32).flatten() / 255.0)

        # GT centroid
        center = parse_centroid(xml_path)
        centroids.append(center if center is not None else (np.nan, np.nan))

    # === PCA computation ===
    X = np.stack(X)
    pca_signal = PCA(n_components=1).fit_transform(X).flatten()

    # === Tracking displacement ===
    centroids = np.array(centroids, dtype=np.float32)
    valid = ~np.isnan(centroids[:, 0])
    centroids = centroids[valid]
    if len(centroids) < 2:
        print(f"Skipping {pid} (not enough valid GT data)")
        continue
    displacement = np.linalg.norm(centroids - centroids[0], axis=1)

    # === Save results ===
    np.save(os.path.join(output_dir, f"{pid}_pca.npy"), pca_signal)
    np.save(os.path.join(output_dir, f"{pid}_tracking.npy"), displacement)
    print(f"Saved PCA and tracking signals for patient {pid}")
