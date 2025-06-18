import torch
print(torch.version.cuda)
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torchvision import transforms
from stenosis_dataset import StenosisDataset
from tqdm import tqdm
import os

# Paths and parameters
train_img_dir = "split_data/train/images"
train_ann_dir = "split_data/train/annotations"
checkpoint_dir = "checkpoints_frcnn_resnet101"
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
model_save_path = "frcnn_resnet101.pth"
num_epochs = 10
batch_size = 4
lr = 0.005

# Make checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)

# Transform and dataset
transform = transforms.ToTensor()
train_dataset = StenosisDataset(train_img_dir, train_ann_dir, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model
backbone = resnet_fpn_backbone("resnet101", weights=ResNet101_Weights.DEFAULT)
model = FasterRCNN(backbone, num_classes=2)  # 1 class + background

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

# Resume from checkpoint
start_epoch = 0
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}")

# Training loop
print("Starting training...")
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/loss_frcnn_resnet101.txt", "a") as f:  # <-- change filename per model
        f.write(f"{epoch_loss:.4f}\n")


    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: epoch {epoch+1}")

# Save final model
torch.save(model.state_dict(), model_save_path)
print(f"Final model saved to: {model_save_path}")
