import csv
from pathlib import Path

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.convnextv2 import convnextv2_atto

ROOT = Path(r"D:\ConvNeXtV2-course-project")
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
])

test_data = datasets.CIFAR100(
    root=ROOT / "data",
    train=False,
    download=False,
    transform=transform,
)
test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

experiments = [
    ("ConvNeXt V1-Atto", ROOT / "runs/formal-v1-atto/best.pth"),
    ("ConvNeXt V2-Atto", ROOT / "runs/formal-v2-atto/best.pth"),
]

results = []

for name, checkpoint_path in experiments:
    if name == "ConvNeXt V1-Atto":
        model = timm.create_model(
            "convnext_atto.d2_in1k",
            pretrained=False,
            num_classes=100,
        )
    else:
        model = convnextv2_atto(num_classes=100)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    model = model.cuda().eval()

    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)

            total_loss += criterion(logits.float(), labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    row = {
        "model": name,
        "best_epoch": checkpoint["epoch"],
        "val_accuracy": checkpoint["val_accuracy"],
        "test_accuracy": correct / total,
        "test_loss": total_loss / total,
        "test_samples": total,
    }
    results.append(row)

    print(
        f"{name}: 最佳轮次={row['best_epoch']}, "
        f"验证准确率={row['val_accuracy']:.2%}, "
        f"测试准确率={row['test_accuracy']:.2%}, "
        f"测试损失={row['test_loss']:.4f}, "
        f"测试图片={total}"
    )

csv_path = OUTPUT / "main_results.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("结果已保存到：", csv_path)
