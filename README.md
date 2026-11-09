# ConvNeXt V2 CIFAR-100 Reproduction

## Project

This project reproduces the ConvNeXt V2 classification model and compares it with ConvNeXt V1-Atto on CIFAR-100.

## Environment

- Python 3.10.11
- PyTorch 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 5060
- Input size: 160x160
- Batch size: 32
- Random seed: 42

## Official Resources

- Paper: https://arxiv.org/abs/2301.00808
- Official repository: https://github.com/facebookresearch/ConvNeXt-V2
- Repository commit: 2553895753323c6fe0b2bf390683f5ea358a42b9

## Main Results

| Model | Validation Accuracy | Test Accuracy | Parameters | Latency | FPS |
|---|---:|---:|---:|---:|---:|
| ConvNeXt V1-Atto | 80.20% | 79.47% | 3.4066M | 4.324 ms | 231.27 |
| ConvNeXt V2-Atto | 80.12% | 79.81% | 3.4195M | 5.596 ms | 178.69 |

## GRN Ablation

Removing GRN from ConvNeXt V2-Atto reduced test accuracy from 79.81% to 74.84%.

## Reproduction Scope

This project verifies the official pretrained model and performs transfer learning on CIFAR-100. Due to laptop hardware limitations, it does not reproduce the complete ImageNet-1K training or FCMAE pretraining process from the original paper.
