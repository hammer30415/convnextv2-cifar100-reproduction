import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from models.convnextv2 import convnextv2_atto

image_path = sys.argv[1]
weight_path = r"D:\ConvNeXtV2-course-project\checkpoints\official\convnextv2_atto_1k_224_ema.pt"

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

model = convnextv2_atto()
checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
model = model.cuda().eval()

image = Image.open(image_path).convert("RGB")
x = transform(image).unsqueeze(0).cuda()
labels = ResNet18_Weights.DEFAULT.meta["categories"]

with torch.inference_mode():
    probabilities = model(x).softmax(dim=1)[0]
    scores, indices = probabilities.topk(5)

for score, index in zip(scores.tolist(), indices.tolist()):
    print(f"{index:4d}  {labels[index]:30s}  {score:.2%}")
