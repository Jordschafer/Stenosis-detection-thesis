import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torchvision.ops import box_iou
from stenosis_dataset import StenosisDataset
from sca_utils import apply_sequence_consistency_alignment

# === CONFIG ===
MODEL_PATH = "frcnn_resnet101.pth"
IMG_DIR = "split_data/test/images"
ANN_DIR = "split_data/test/annotations"
OUTPUT_DIR = "logs/predictions/frcnn_resnet101_frames"
EVAL_XML = "evaluation_results_frcnn_resnet101_sca_per_frame.xml"
IOU_LOG = "logs/ious_frcnn_resnet101_sca_per_frame.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# === MODEL ===
backbone = resnet_fpn_backbone("resnet101", weights=ResNet101_Weights.DEFAULT)
model = FasterRCNN(backbone, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === DATA ===
transform = transforms.ToTensor()
dataset = StenosisDataset(IMG_DIR, ANN_DIR, transforms=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# === INFERENCE ===
print("🔍 Running inference...")
frame_infos = []  # To track all frame metadata
raw_detections_per_patient = defaultdict(list)
gt_boxes_per_patient = defaultdict(list)
frame_names_per_patient = defaultdict(list)
image_filenames = sorted(os.listdir(IMG_DIR))

for idx, (images, targets) in tqdm(enumerate(loader), total=len(loader)):
    image = images[0].to(device)
    target = targets[0]

    with torch.no_grad():
        preds = model([image])[0]

    # Get frame info
    img_filename = image_filenames[idx]
    patient_id = "_".join(img_filename.split("_")[:3])

    # Group by patient
    raw_detections_per_patient[patient_id].append({
        "boxes": preds["boxes"].cpu(),
        "scores": preds["scores"].cpu(),
        "labels": preds["labels"].cpu()
    })
    gt_boxes_per_patient[patient_id].append(target["boxes"])
    frame_names_per_patient[patient_id].append(img_filename)

    frame_infos.append((patient_id, img_filename))  # Save for later alignment

# === APPLY SCA PER PATIENT ===
print("🧠 Applying SCA per patient...")
filtered_detections_all = {}
gt_boxes_all = {}

for pid in tqdm(raw_detections_per_patient):
    detections = raw_detections_per_patient[pid]
    filtered = apply_sequence_consistency_alignment(detections)
    frames = frame_names_per_patient[pid]
    gts = gt_boxes_per_patient[pid]

    for fname, det, gt in zip(frames, filtered, gts):
        filtered_detections_all[fname] = det
        gt_boxes_all[fname] = gt

# === SAVE PREDICTED BOXES PER FRAME ===
last_valid_box = None

for fname, det in zip(frames, filtered):  # one frame at a time for this patient
    if len(det["boxes"]) > 0:
        # use first box, update memory
        best_box = det["boxes"][0].cpu().numpy().tolist()
        last_valid_box = best_box
    else:
        # use previous box if available
        best_box = last_valid_box
        if best_box is not None:
            print(f"ℹ️ Using last valid box for {fname}")
        else:
            print(f"⚠️ No box and no fallback available for {fname}")

    out_path = os.path.join(OUTPUT_DIR, f"{fname}_boxes.npy")
    np.save(out_path, best_box)

# === EVALUATE ===
print("📊 Evaluating predictions...")
TP, FP, FN = 0, 0, 0
ious = []

for fname in sorted(filtered_detections_all.keys()):
    pred_boxes = filtered_detections_all[fname]["boxes"]
    gt_boxes = gt_boxes_all[fname].to(pred_boxes.device)

    if len(pred_boxes) == 0 and len(gt_boxes) > 0:
        FN += len(gt_boxes)
        continue
    elif len(gt_boxes) == 0:
        FP += len(pred_boxes)
        continue

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    for row in iou_matrix:
        max_iou, gt_idx = row.max(0)
        if max_iou >= 0.5 and gt_idx.item() not in matched_gt:
            TP += 1
            ious.append(max_iou.item())
            matched_gt.add(gt_idx.item())
        else:
            FP += 1

    FN += len(gt_boxes) - len(matched_gt)

# === METRICS ===
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
mean_iou = sum(ious) / len(ious) if ious else 0

print("\n=== Evaluation Results (SCA, per-frame saved) ===")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}, mIoU: {mean_iou:.3f}")

with open(IOU_LOG, "w") as f:
    for iou in ious:
        f.write(f"{iou:.4f}\n")

# === SAVE XML ===
def save_results_to_xml(results, filename):
    root = ET.Element("evaluation")
    for key, value in results.items():
        ET.SubElement(root, key.replace(" ", "_")).text = str(value)
    ET.ElementTree(root).write(filename)

save_results_to_xml({
    "True Positives": TP,
    "False Positives": FP,
    "False Negatives": FN,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "Mean IoU": mean_iou
}, EVAL_XML)
