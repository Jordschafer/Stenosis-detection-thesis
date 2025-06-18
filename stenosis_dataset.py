import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image

class StenosisDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(".bmp")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + ".XML")

        img = Image.open(img_path).convert("RGB")  # convert to 3-channel
        boxes, labels = self.parse_voc_xml(ann_path)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def parse_voc_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = 1  # 'Stenosis' = class 1
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels
