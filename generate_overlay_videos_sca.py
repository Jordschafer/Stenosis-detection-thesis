import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import OrderedDict
from torchvision.transforms import ToTensor
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from sca_utils import apply_sequence_consistency_alignment
from torch import nn

# === CONFIG ===
IMAGE_DIR = "split_data/test/images"
ANNOTATION_DIR = "split_data/test/annotations"
OUTPUT_DIR = "videos_sca_live"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Helper: Load ground truth boxes ===
def load_ground_truth(xml_path):
    boxes = []
    if not os.path.exists(xml_path):
        return boxes
    try:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))
            boxes.append((xmin, ymin, xmax, ymax))
    except Exception as e:
        print(f"XML parse error: {e}")
    return boxes

# === Model Definitions ===
def load_frcnn_resnet101():
    backbone = resnet_fpn_backbone("resnet101", weights=ResNet101_Weights.DEFAULT)
    model = FasterRCNN(backbone, num_classes=2)
    model.load_state_dict(torch.load("frcnn_resnet101.pth", map_location=DEVICE))
    return model.eval().to(DEVICE)

def load_frcnn_inceptionv3():
    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights, aux_logits=True)
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

    class InceptionBackbone(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.body = backbone
            self.out_channels = 768

        def forward(self, x):
            return {"0": self.body(x)}

    inception_backbone = InceptionBackbone(backbone)

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=inception_backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    model.load_state_dict(torch.load("frcnn_inceptionv3.pth", map_location=DEVICE))
    return model.eval().to(DEVICE)

MODELS = {
    "FRCNN_ResNet101": load_frcnn_resnet101,
    "FRCNN_InceptionV3": load_frcnn_inceptionv3
}

# === Generate video for one model
def generate_overlay_video(model_name, model_fn):
    model = model_fn()
    frame_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".bmp")])
    raw_detections = []

    print(f"🔍 Running inference for {model_name}...")
    for fname in tqdm(frame_files, desc="Inference"):
        image = cv2.imread(os.path.join(IMAGE_DIR, fname))
        img_tensor = ToTensor()(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = model(img_tensor)[0]

        raw_detections.append({
            "boxes": preds["boxes"].to(DEVICE),
            "scores": preds["scores"].to(DEVICE),
            "labels": preds["labels"].to(DEVICE)
        })

    print("🧠 Applying SCA...")
    filtered_detections = apply_sequence_consistency_alignment(
        raw_detections,
        t_iou=0.3,
        t_frame=3,
        t_score_interp=0.1,
        max_frame_gap_for_linking=1,
        debug=False
    )

    print("🎥 Generating video...")
    h, w = cv2.imread(os.path.join(IMAGE_DIR, frame_files[0])).shape[:2]
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_with_sca.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))

    for i, fname in enumerate(tqdm(frame_files, desc="Rendering")):
        frame = cv2.imread(os.path.join(IMAGE_DIR, fname))
        xml_path = os.path.join(ANNOTATION_DIR, os.path.splitext(fname)[0] + ".XML")

        # Draw GT (green)
        for x1, y1, x2, y2 in load_ground_truth(xml_path):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw SCA-filtered prediction (red)
        for box in filtered_detections[i]["boxes"]:
            x1, y1, x2, y2 = map(int, box.cpu().tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(frame, f"{model_name} | {fname}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()
    print(f"✅ Saved: {out_path}")

# === Run for all models
if __name__ == "__main__":
    for model_name, model_fn in MODELS.items():
        generate_overlay_video(model_name, model_fn)
