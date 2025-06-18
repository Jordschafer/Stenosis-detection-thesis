import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import euclidean

# === CONFIGURATION ===
models = ["frcnn_resnet101_frames", "frcnn_inceptionv3_frames"]
prediction_dir = "logs/predictions"
output_dir = "logs/patient_signals_pred"
os.makedirs(output_dir, exist_ok=True)

def extract_tracking_signal(boxes_per_frame, frame_names):
    centers = []
    buffer = []

    for i, box in enumerate(boxes_per_frame):
        print(f"\n📦 Frame {frame_names[i]}: loaded → {box}")
            
        if box is None:
            print("   ⛔ Skipped: box is None")
            centers.append([np.nan, np.nan])
            continue

        # Full box with 4 values
        if isinstance(box, (list, np.ndarray)):
            try:
                if len(box) == 4:
                    xmin, ymin, xmax, ymax = box
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    print(f"   ✅ Parsed box → center: ({cx:.2f}, {cy:.2f})")
                    centers.append([cx, cy])
                    continue
            except Exception as e:
                print(f"   ⚠️ Error parsing full box: {box} → {e}")
                centers.append([np.nan, np.nan])
                continue

        if isinstance(box, (list, np.ndarray)) and hasattr(box, '__len__') and len(box) == 4:
            xmin, ymin, xmax, ymax = box
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            print(f"   ✅ Parsed box → center: ({cx:.2f}, {cy:.2f})")
            centers.append([cx, cy])

        elif isinstance(box, (float, np.floating, int)):
            buffer.append(float(box))
            if len(buffer) == 4:
                xmin, ymin, xmax, ymax = buffer
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                centers.append([cx, cy])
                print(f"   🔄 Reconstructed box → center: ({cx:.2f}, {cy:.2f})")
                buffer.clear()
        else:
            print(f"   ⚠️ Unrecognized format: {box}")
            centers.append([np.nan, np.nan])

    if buffer:
        print(f"   ⚠️ Leftover scalars: {buffer}")

    centers = np.array(centers, dtype=np.float32)

    # Interpolate x and y separately
    df = pd.DataFrame(centers, columns=["cx", "cy"])
    df_interp = df.interpolate(method="linear", limit_direction="both")

    # Generate motion signal from center movement
    signal = []
    ref = None
    for cx, cy in df_interp.values:
        if np.isnan(cx) or np.isnan(cy):
            signal.append(np.nan)
        elif ref is None:
            ref = (cx, cy)
            signal.append(0.0)
        else:
            signal.append(euclidean((cx, cy), ref))

    return np.array(signal, dtype=np.float32)

# === MAIN SCRIPT ===
for model in models:
    model_input_dir = os.path.join(prediction_dir, model)
    model_output_dir = os.path.join(output_dir, model)
    os.makedirs(model_output_dir, exist_ok=True)

    patient_frames = {}

    for fname in sorted(os.listdir(model_input_dir)):
        if not fname.endswith("_boxes.npy"):
            continue
        patient_id = "_".join(fname.split("_")[:3])
        patient_frames.setdefault(patient_id, []).append(fname)

    for pid, frame_list in patient_frames.items():
        boxes_per_frame = []
        frame_names = []

        for frame_file in sorted(frame_list):
            path = os.path.join(model_input_dir, frame_file)
            try:
                box = np.load(path, allow_pickle=True)
                boxes_per_frame.append(box)
                frame_names.append(frame_file.replace("_boxes.npy", ""))
            except Exception as e:
                print(f"[!] Failed to load {frame_file}: {e}")
                boxes_per_frame.append(None)
                frame_names.append(frame_file)

        print(f"\n🔍 Extracting tracking signal for {model}/{pid} ({len(frame_names)} frames)")
        signal = extract_tracking_signal(boxes_per_frame, frame_names)

        print(f"\n📊 Signal summary for {pid}:")
        print(f"   ➤ Length: {len(signal)}")
        print(f"   ➤ NaNs: {np.count_nonzero(np.isnan(signal))}")
        print(f"   ➤ Zeros: {np.count_nonzero(signal == 0.0)}")
        print(f"   ➤ Min: {np.nanmin(signal)}, Max: {np.nanmax(signal)}")

        out_path = os.path.join(model_output_dir, f"{pid}_tracking.npy")
        np.save(out_path, signal)
        print(f"✅ Saved → {out_path}")
