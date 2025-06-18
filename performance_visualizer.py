import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# === CONFIG ===
EVAL_FILES = {
    'SSD + MobileNetV3': 'evaluation_results_ssd_sca.xml',
    'FRCNN + ResNet101': 'evaluation_results_frcnn_resnet101_sca.xml',
    'FRCNN + InceptionV3': 'evaluation_results_frcnn_inceptionv3.xml'
}
LOSS_LOGS = {
    'SSD + MobileNetV3': 'logs/loss_ssd.txt',
    'FRCNN + ResNet101': 'logs/loss_resnet101.txt',
    'FRCNN + InceptionV3': 'logs/loss_inceptionv3.txt'
}
IOU_LOGS = {
    'SSD + MobileNetV3': 'logs/ious_ssd.txt',
    'FRCNN + ResNet101': 'logs/ious_resnet101.txt',
    'FRCNN + InceptionV3': 'logs/ious_inceptionv3.txt'
}

# === HELPER FUNCTIONS ===
def parse_eval_xml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    return {
        "Precision": float(root.findtext("Precision", 0)),
        "Recall": float(root.findtext("Recall", 0)),
        "F1-score": float(root.findtext("F1-score", 0)),
        "Mean IoU": float(root.findtext("Mean IoU", 0))
    }

def parse_loss_log(filepath):
    losses = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                losses.append(float(line.strip()))
            except ValueError:
                continue
    return losses

def parse_ious(filepath):
    ious = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                ious.append(float(line.strip()))
            except ValueError:
                continue
    return ious

# === 1. BAR CHARTS OF METRICS ===
metrics = {}
for model_name, xml_file in EVAL_FILES.items():
    if os.path.exists(xml_file):
        metrics[model_name] = parse_eval_xml(xml_file)

plt.figure(figsize=(10, 6))
for metric in ["Precision", "Recall", "F1-score", "Mean IoU"]:
    plt.clf()
    for model_name, result in metrics.items():
        plt.bar(model_name, result[metric])
    plt.title(metric)
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{metric.lower().replace(' ', '_')}_comparison.png")
    plt.show()

# === 2. TRAINING LOSS CURVES ===
plt.figure(figsize=(10, 6))
for model_name, log_path in LOSS_LOGS.items():
    if os.path.exists(log_path):
        loss = parse_loss_log(log_path)
        plt.plot(range(1, len(loss)+1), loss, label=model_name)
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("training_loss_comparison.png")
plt.show()

# === 3. IoU DISTRIBUTIONS ===
plt.figure(figsize=(10, 6))
for model_name, iou_file in IOU_LOGS.items():
    if os.path.exists(iou_file):
        ious = parse_ious(iou_file)
        plt.hist(ious, bins=30, alpha=0.6, label=model_name, density=True)
plt.title("IoU Distribution per Model")
plt.xlabel("IoU")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("iou_distribution_comparison.png")
plt.show()
