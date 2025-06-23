import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torch import nn
from collections import OrderedDict
from stenosis_dataset import StenosisDataset
from sca_utils import apply_sequence_consistency_alignment

# === CONFIG ===
MODEL_PATH = "frcnn_inceptionv3.pth"
IMG_DIR = "split_data/test/images"
ANN_DIR = "split_data/test/annotations"
OUTPUT_DIR = "logs/predictions/frcnn_inceptionv3_frames"
EVAL_XML = "evaluation_results_frcnn_inceptionv3_sca_persist.xml"
IOU_LOG = "logs/ious_frcnn_inceptionv3_sca_persist.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# === MODEL ===
inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
inception.aux_logits = False
backbone = nn.Sequential(OrderedDict([
    ('Conv2d_1a_3x3', inception.Conv2d_1a_3x3),
    ('Conv2d_2a_3x3', inception.Conv2d_2a_3x3),
    ('Conv2d_2b_3x3', inception.Conv2d_2b_3x3),
    ('maxpool1', inception.maxpool1),
    ('Conv2d_3b_1x1', inception.Conv2d_3b_1x1),
    ('Conv2d_4a_3x3', inception.Conv2d_4a_3x3),
    ('maxpool2', inception.maxpool2),
    ('Mixed_5b', inception.Mixed_5b),
    ('Mixed_5c', inception.Mixed_5c),
    ('Mixed_5d', inception.Mixed_5d),
    ('Mixed_6a', inception.Mixed_6a),
    ('Mixed_6b', inception.Mixed_6b),
    ('Mixed_6c', inception.Mixed_6c),
    ('Mixed_6d', inception.Mixed_6d),
    ('Mixed_6e', inception.Mixed_6e),
]))

class InceptionBackbone(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        self.out_channels = 768
    def forward(self, x):
        return {"0": self.body(x)}

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

model = FasterRCNN(
    backbone=InceptionBackbone(backbone),
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === DATA ===
transform = transforms.ToTensor()
dataset = StenosisDataset(IMG_DIR, ANN_DIR, transforms=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
image_filenames = sorted(os.listdir(IMG_DIR))

# === INFERENCE ===
raw_detections_per_patient = defaultdict(list)
gt_boxes_per_patient = defaultdict(list)
frame_names_per_patient = defaultdict(list)
frame_infos = []

for idx, (images, targets) in tqdm(enumerate(loader), total=len(loader), desc="🔍 Running Inference"):
    image = images[0].to(device)
    target = targets[0]
    with torch.no_grad():
        preds = model([image])[0]

    img_filename = image_filenames[idx]
    patient_id = "_".join(img_filename.split("_")[:3])

    raw_detections_per_patient[patient_id].append({
        "boxes": preds["boxes"].cpu(),
        "scores": preds["scores"].cpu(),
        "labels": preds["labels"].cpu()
    })
    gt_boxes_per_patient[patient_id].append(target["boxes"])
    frame_names_per_patient[patient_id].append(img_filename)
    frame_infos.append((patient_id, img_filename))

# === SCA per patient
filtered_detections_all = {}
gt_boxes_all = {}

for pid in tqdm(raw_detections_per_patient, desc=" Applying SCA"):
    detections = raw_detections_per_patient[pid]
    filtered = apply_sequence_consistency_alignment(detections)
    frames = frame_names_per_patient[pid]
    gts = gt_boxes_per_patient[pid]

    last_valid_box = None

    for fname, det, gt in zip(frames, filtered, gts):
        boxes = det["boxes"]
        best_box = None

        if len(boxes) > 0:
            if boxes.ndim == 2 and boxes.shape[1] == 4:
                best_box = boxes[0].cpu().numpy().tolist()
            elif boxes.ndim == 1 and boxes.shape[0] == 4:
                best_box = boxes.cpu().numpy().tolist()

        if best_box is None and last_valid_box is not None:
            best_box = last_valid_box
            print(f" {fname}: no prediction, reused last valid box")

        if best_box is not None:
            last_valid_box = best_box

        # Save prediction (even if best_box is None)
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_boxes.npy")
        np.save(out_path, best_box)

        filtered_detections_all[fname] = {"boxes": torch.tensor([best_box]) if best_box else torch.empty((0, 4))}
        gt_boxes_all[fname] = gt

# === EVALUATION ===
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

print("\n=== Evaluation Results ===")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}, mIoU: {mean_iou:.3f}")

with open(IOU_LOG, "w") as f:
    for iou in ious:
        f.write(f"{iou:.4f}\n")

# === SAVE XML
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
