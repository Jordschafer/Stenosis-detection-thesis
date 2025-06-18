import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Pad naar jouw datafolder (pas dit aan indien nodig)
DATA_DIR = Path("Data/Stenosis detection/dataset")
OUTPUT_DIR = Path("split_data")

# Output folders
TRAIN_IMG_DIR = OUTPUT_DIR / "train" / "images"
TRAIN_ANN_DIR = OUTPUT_DIR / "train" / "annotations"
TEST_IMG_DIR = OUTPUT_DIR / "test" / "images"
TEST_ANN_DIR = OUTPUT_DIR / "test" / "annotations"

# Maak output directories aan
for folder in [TRAIN_IMG_DIR, TRAIN_ANN_DIR, TEST_IMG_DIR, TEST_ANN_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Verzamel alle BMP-bestanden
bmp_files = list(DATA_DIR.glob("*.BMP"))
print(f"Gevonden {len(bmp_files)} BMP-afbeeldingen...")

# Groepeer bestanden per patiënt-ID
# (Bestandsnaam: 14_002_5_0016.BMP → patiënt-ID = 002)
patient_files = defaultdict(list)
for bmp in bmp_files:
    name_parts = bmp.stem.split("_")
    if len(name_parts) >= 3:
        patient_id = name_parts[1]  # bijv. "002"
        patient_files[patient_id].append(bmp)

# Train/test split op basis van patiënt
patient_ids = list(patient_files.keys())
random.seed(42)
random.shuffle(patient_ids)
split_idx = int(0.8 * len(patient_ids))
train_ids = set(patient_ids[:split_idx])
test_ids = set(patient_ids[split_idx:])

# Hulpfunctie om bestanden te kopiëren
def copy_files(patient_id_set, img_out, ann_out):
    for pid in patient_id_set:
        for bmp_path in patient_files[pid]:
            xml_path = bmp_path.with_suffix(".XML")
            if xml_path.exists():
                shutil.copy(bmp_path, img_out / bmp_path.name)
                shutil.copy(xml_path, ann_out / xml_path.name)

# Bestanden kopiëren
copy_files(train_ids, TRAIN_IMG_DIR, TRAIN_ANN_DIR)
copy_files(test_ids, TEST_IMG_DIR, TEST_ANN_DIR)

# Stats printen
print("\n Split klaar")
print(f"Train: {len(train_ids)} patiënten, {len(list(TRAIN_IMG_DIR.glob('*.BMP')))} afbeeldingen")
print(f"Test : {len(test_ids)} patiënten, {len(list(TEST_IMG_DIR.glob('*.BMP')))} afbeeldingen")
