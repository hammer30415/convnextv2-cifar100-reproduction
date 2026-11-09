import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.convnextv2 import convnextv2_atto
from models.utils import GRN

ROOT = Path(r"D:\ConvNeXtV2-course-project")
WEIGHT = ROOT / "checkpoints/official/convnextv2_atto_1k_224_ema.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--max-train", type=int, default=512)
parser.add_argument("--max-val", type=int, default=256)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

args.output.mkdir(parents=True, exist_ok=True)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

with open(ROOT / "data/cifar100_split_seed42.json", encoding="utf-8") as f:
    split = json.load(f)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_data = datasets.CIFAR100(ROOT / "data", train=True, download=False, transform=train_tf)
val_data = datasets.CIFAR100(ROOT / "data", train=True, download=False, transform=val_tf)

train_ids = split["train_indices"][:args.max_train]
val_ids = split["val_indices"][:args.max_val]

train_loader = DataLoader(
    Subset(train_data, train_ids),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
val_loader = DataLoader(
    Subset(val_data, val_ids),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

device = "cuda"
model = convnextv2_atto(num_classes=1000)
checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
replaced = 0
for block in model.modules():
    if hasattr(block, "grn") and isinstance(block.grn, GRN):
        block.grn = nn.Identity()
        replaced += 1
print("??? GRN ???:", replaced)
assert replaced > 0
model.head = nn.Linear(model.head.in_features, 100)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")

history_path = args.output / "history.csv"
best_accuracy = -1.0

with open(history_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * labels.size(0)

        model.eval()
        val_loss_sum = 0.0
        correct = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()

        train_loss = train_loss_sum / len(train_ids)
        val_loss = val_loss_sum / len(val_ids)
        val_accuracy = correct / len(val_ids)
        writer.writerow([epoch, train_loss, val_loss, val_accuracy])
        f.flush()

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.2%}",
            flush=True,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "seed": seed,
                    "image_size": 160,
                },
                args.output / "best.pth",
            )

print("完成；结果目录：", args.output)
