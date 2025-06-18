import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from stenosis_dataset import StenosisDataset
from tqdm import tqdm

# Parameters from paper 
INPUT_SIZE = (300, 300)  # paper used 300x300 for SSD MobileNet V2
BATCH_SIZE = 4
LEARNING_RATE = 0.004
NUM_EPOCHS = 10
MODEL_PATH = "ssd_mobilenetv2_stenosis.pth"

# Data paths
train_img_dir = os.path.join("split_data", "train", "images")
train_ann_dir = os.path.join("split_data", "train", "annotations")

def ssd_random_crop(img):
    # Placeholder for actual SSD-style random crop
    return transforms.RandomCrop(INPUT_SIZE)(img)  # Simplified stand-in

transform_pipeline = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(ssd_random_crop),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
train_dataset = StenosisDataset(train_img_dir, train_ann_dir, transforms=transform_pipeline)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          collate_fn=lambda x: tuple(zip(*x)))

# Model: SSD MobileNetV2
model = ssdlite320_mobilenet_v3_large(weights=None)
model.head.classification_head.num_classes = 2

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Training Loop
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/loss_ssd.txt", "a") as f:  # <-- change filename per model
        f.write(f"{total_loss:.4f}\n")

# Save Trained Model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")