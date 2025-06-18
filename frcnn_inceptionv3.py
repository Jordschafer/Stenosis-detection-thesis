import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms
from stenosis_dataset import StenosisDataset
from collections import OrderedDict
from torch import nn
import os
from tqdm import tqdm

class InceptionBackbone(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.body = backbone
        self.out_channels = 768  # channels from Mixed_6e

    def forward(self, x):
        x = self.body(x)
        return {"0": x}

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Inception v3 backbone
inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
inception.aux_logits = False

# Modify the backbone
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
inception_backbone = InceptionBackbone(backbone)

# Anchor generator and ROI pooler
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# Build model
model = FasterRCNN(backbone=inception_backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
model.to(device)

# Dataset and loader
transform = transforms.ToTensor()
train_img_dir = "split_data/train/images"
train_ann_dir = "split_data/train/annotations"
train_dataset = StenosisDataset(train_img_dir, train_ann_dir, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Checkpointing
checkpoint_dir = "checkpoints_frcnn_inceptionv3"
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
os.makedirs(checkpoint_dir, exist_ok=True)

start_epoch = 0
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}")

# Training loop
num_epochs = 10
print("Starting training...")
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/loss_frcnn_inceptionv3.txt", "a") as f:  # <-- change filename per model
        f.write(f"{total_loss:.4f}\n")


    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved for epoch {epoch+1}")

# Final save
torch.save(model.state_dict(), "frcnn_inceptionv3.pth")
print("Final model saved to frcnn_inceptionv3.pth")