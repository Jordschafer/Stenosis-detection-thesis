import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# === CONFIGURATION ===
models = ["frcnn_resnet101_frames", "frcnn_inceptionv3_frames"]
input_base_dir = "logs/patient_signals_pred"
output_base_dir = "logs/patient_signals_pred_combined"
os.makedirs(output_base_dir, exist_ok=True)

for model in models:
    print(f"\n Processing model: {model}")
    input_dir = os.path.join(input_base_dir, model)
    output_dir = os.path.join(output_base_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    patient_signals = defaultdict(list)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_tracking.npy")])

    for fname in tqdm(frame_files, desc=f" Reading {model}"):
        full_path = os.path.join(input_dir, fname)
        frame_name = fname.replace("_tracking.npy", "")
        patient_id = "_".join(frame_name.split("_")[:3])

        try:
            raw = np.load(full_path, allow_pickle=True)
            print(f"\n {fname}: loaded → {raw}")

            # Determine scalar value
            if isinstance(raw, (float, int)):
                value = float(raw)
            elif isinstance(raw, np.ndarray):
                if raw.ndim == 0:
                    value = float(raw)
                elif raw.size == 1:
                    value = float(raw.flatten()[0])
                else:
                    print(f"⚠ Skipped {fname}: array with size > 1 → {raw}")
                    continue
            else:
                print(f" Skipped {fname}: unsupported type {type(raw)}")
                continue

            print(f" patient: {patient_id}, frame: {frame_name}, signal: {value}")
            patient_signals[patient_id].append((frame_name, value))

        except Exception as e:
            print(f" Failed to load {fname}: {e}")

    # === COMBINE PER PATIENT ===
    for pid, items in patient_signals.items():
        items.sort(key=lambda x: x[0])  # sort by frame name

        signal_sequence = []
        for frame, sig in items:
            signal_sequence.append(sig)

        signal_array = np.array(signal_sequence, dtype=np.float32)

        # Summary diagnostics
        total = len(signal_array)
        n_nans = np.count_nonzero(np.isnan(signal_array))
        n_zeros = np.count_nonzero(signal_array == 0.0)

        print(f"\n {model} - {pid}:")
        print(f"   Total frames: {total}")
        print(f"   NaNs: {n_nans}")
        print(f"   Zeros: {n_zeros}")
        print(f"   Min: {np.nanmin(signal_array) if total > n_nans else 'NaN'}, Max: {np.nanmax(signal_array) if total > n_nans else 'NaN'}")

        if n_nans == total:
            print(f" Skipped saving {pid}: signal is entirely NaN")
            continue

        # Save
        out_path = os.path.join(output_dir, f"{pid}_tracking.npy")
        np.save(out_path, signal_array)
        print(f" Saved: {out_path}")
