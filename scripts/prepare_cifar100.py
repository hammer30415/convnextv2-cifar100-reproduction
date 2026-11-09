import json
from pathlib import Path

import torch
from torchvision.datasets import CIFAR100

root = Path(r"D:\ConvNeXtV2-course-project")
data_dir = root / "data"
data_dir.mkdir(exist_ok=True)

train_data = CIFAR100(root=data_dir, train=True, download=True)
test_data = CIFAR100(root=data_dir, train=False, download=True)

indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(42))
split = {
    "seed": 42,
    "train_indices": indices[5000:].tolist(),
    "val_indices": indices[:5000].tolist(),
}
output = data_dir / "cifar100_split_seed42.json"
output.write_text(json.dumps(split), encoding="utf-8")

print("训练集:", len(split["train_indices"]))
print("验证集:", len(split["val_indices"]))
print("官方测试集:", len(test_data))
print("划分文件:", output)
