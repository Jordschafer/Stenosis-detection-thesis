import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import box_iou
from stenosis_dataset import StenosisDataset
from sca_utils import apply_sequence_consistency_alignment
import xml.etree.ElementTree as ET
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paths
MODEL_PATH = "ssd_mobilenetv2_stenosis.pth"
TEST_IMG_DIR = "split_data/test/images"
TEST_ANN_DIR = "split_data/test/annotations"

# Transform
transform_pipeline = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# Load model
model = ssdlite320_mobilenet_v3_large(weights=None)
model.head.classification_head.num_classes = 2  # background + stenosis
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)  # <-- Move model to GPU if available
model.eval()

# Load test data
test_dataset = StenosisDataset(TEST_IMG_DIR, TEST_ANN_DIR, transforms=transform_pipeline)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Collect predictions
raw_detections = []
gt_boxes_all = []

for images, targets in test_loader:
    image = images[0].to(device)  # <-- Move image to GPU
    target = targets[0]

    with torch.no_grad():
        preds = model([image])[0]

    raw_detections.append({
        "boxes": preds['boxes'],
        "scores": preds['scores'],
        "labels": preds['labels']
    })
    gt_boxes_all.append(target['boxes'])

# Apply SCA post-processing
filtered_detections = apply_sequence_consistency_alignment(
    raw_detections,
    t_iou=0.3,
    t_frame=3,
    t_score_interp=0.1,
    max_frame_gap_for_linking=1,
    debug=False
)

# Evaluation counters
TP, FP, FN = 0, 0, 0
iou_threshold = 0.5
ious = []

for det, gt_boxes in zip(filtered_detections, gt_boxes_all):
    pred_boxes = det['boxes']
    gt_boxes = gt_boxes.to(pred_boxes.device)

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
        if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
            TP += 1
            ious.append(max_iou.item())
            matched_gt.add(gt_idx.item())
        else:
            FP += 1

    FN += len(gt_boxes) - len(matched_gt)

# Metrics
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
mean_iou = sum(ious) / len(ious) if ious else 0

# Print results
print("=== SSD Evaluation Results with SCA ===")
print(f"True Positives : {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Precision      : {precision:.3f}")
print(f"Recall         : {recall:.3f}")
print(f"F1-score       : {f1:.3f}")
print(f"Mean IoU       : {mean_iou:.3f}")

os.makedirs("logs", exist_ok=True)
with open("logs/ious_ssd.txt", "w") as f:
    for iou in ious:
        f.write(f"{iou:.4f}\n")

# Save results to XML
def save_results_to_xml(results, filename):
    root = ET.Element("evaluation")
    for key, value in results.items():
        tag = key.replace(" ", "_")  # <-- sanitize tag names
        ET.SubElement(root, tag).text = str(value)
    ET.ElementTree(root).write(filename)

save_results_to_xml({
    "True Positives": TP,
    "False Positives": FP,
    "False Negatives": FN,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "Mean IoU": mean_iou
}, "evaluation_results_ssd_sca.xml")
