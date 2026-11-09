from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.convnextv2 import convnextv2_atto
from models.utils import GRN

ROOT = Path(r"D:\ConvNeXtV2-course-project")

tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

data = datasets.CIFAR100(ROOT/"data", train=False, download=False, transform=tf)
loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=0)

model = convnextv2_atto(num_classes=100)
for block in model.modules():
    if hasattr(block, "grn") and isinstance(block.grn, GRN):
        block.grn = torch.nn.Identity()

ckpt = torch.load(
    ROOT/"runs/ablation-v2-no-grn/best.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(ckpt["model"])
model = model.cuda().eval()

correct = 0
with torch.inference_mode():
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()

print("最佳轮次:", ckpt["epoch"])
print("验证准确率:", f"{ckpt['val_accuracy']:.2%}")
print("去除 GRN 测试准确率:", f"{correct/len(data):.2%}")
print("测试图片:", len(data))
