import csv
import statistics
import time
from pathlib import Path

import timm
import torch
from models.convnextv2 import convnextv2_atto

ROOT = Path(r"D:\ConvNeXtV2-course-project")
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)

models = [
    ("ConvNeXt V1-Atto", timm.create_model(
        "convnext_atto.d2_in1k", pretrained=False, num_classes=100)),
    ("ConvNeXt V2-Atto", convnextv2_atto(num_classes=100)),
]

rows = []
for name, model in models:
    params = sum(p.numel() for p in model.parameters())
    model = model.cuda().eval()
    x = torch.randn(1, 3, 160, 160, device="cuda")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(30):
            model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    latency = statistics.mean(times)
    row = {
        "model": name,
        "parameters_m": round(params / 1_000_000, 4),
        "latency_ms": round(latency, 4),
        "fps_batch1": round(1000 / latency, 2),
        "input_size": 160,
        "batch_size": 1,
        "precision": "FP16 autocast",
    }
    rows.append(row)
    print(row)
    del model
    torch.cuda.empty_cache()

path = OUTPUT / "efficiency_results.csv"
with path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("结果已保存到：", path)
