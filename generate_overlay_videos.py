import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torchvision.transforms import ToTensor
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm
from collections import OrderedDict
from torch import nn

from stenosis_dataset import StenosisDataset

# === Paths and Setup ===
SPLITS = ["train", "test"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Pascal VOC Ground Truth ===
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
        print(f"Error parsing {xml_path}: {e}")
    return boxes

# === Model Loaders ===
def load_ssd():
    model = ssdlite320_mobilenet_v3_large(weights=None)
    model.head.classification_head.num_classes = 2
    model.load_state_dict(torch.load("ssd_mobilenetv2_stenosis.pth", map_location=DEVICE))
    return model.eval().to(DEVICE)

def load_frcnn_resnet101():
    backbone = resnet_fpn_backbone("resnet101", weights=ResNet101_Weights.DEFAULT)
    model = FasterRCNN(backbone, num_classes=2)
    model.load_state_dict(torch.load("frcnn_resnet101.pth", map_location=DEVICE))
    return model.eval().to(DEVICE)

def load_frcnn_inceptionv3():
    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights, aux_logits=True)
    inception.aux_logits = False

    # Truncate Inception v3 at Mixed_6e as in training
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
        def __init__(self, backbone):
            super().__init__()
            self.body = backbone
            self.out_channels = 768

        def forward(self, x):
            return {"0": self.body(x)}

    inception_backbone = InceptionBackbone(backbone)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=inception_backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    model.load_state_dict(torch.load("frcnn_inceptionv3.pth", map_location=DEVICE))
    return model.eval().to(DEVICE)

MODELS = {
    "SSD_MobileNetV3": load_ssd,
    "FRCNN_ResNet101": load_frcnn_resnet101,
    "FRCNN_InceptionV3": load_frcnn_inceptionv3
}

# === Video Generator ===
def create_video(model_name, model_loader):
    model = model_loader()

    for split in SPLITS:
        print(f"\nProcessing {model_name} on {split} set")
        dataset = StenosisDataset(f"split_data/{split}/images", f"split_data/{split}/annotations", transforms=ToTensor())
        patient_ids = sorted({img.split("_")[1] for img in dataset.images})
        patient_id = patient_ids[0]
        patient_frames = [f for f in dataset.images if f.split("_")[1] == patient_id]

        example_img = cv2.imread(os.path.join(f"split_data/{split}/images", patient_frames[0]))
        height, width = example_img.shape[:2]
        out_path = f"{OUTPUT_DIR}/{model_name}_{split}.mp4"
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))

        for frame_name in tqdm(patient_frames, desc=f"{model_name} - {split}"):
            img_path = os.path.join(f"split_data/{split}/images", frame_name)
            xml_path = os.path.join(f"split_data/{split}/annotations", os.path.splitext(frame_name)[0] + ".XML")
            frame = cv2.imread(img_path)
            img_tensor = ToTensor()(frame).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                preds = model(img_tensor)[0]

            # Draw predictions (red)
            for box in preds['boxes']:
                x1, y1, x2, y2 = map(int, box.cpu().tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw ground truth (green)
            for x1, y1, x2, y2 in load_ground_truth(xml_path):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Overlay text
            cv2.putText(frame, f"{model_name} | {split} | {frame_name}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            video_writer.write(frame)

        video_writer.release()
        print(f"Saved: {out_path}")

# === Run All Models ===
if __name__ == "__main__":
    for model_name, loader in MODELS.items():
        create_video(model_name, loader)
