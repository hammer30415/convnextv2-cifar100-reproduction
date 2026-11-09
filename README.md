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

## Trained checkpoints

Download the three CIFAR-100 checkpoints from [Release v1.0](https://github.com/hammer30415/convnextv2-cifar100-reproduction/releases/tag/v1.0).

| Model | Best epoch | Validation accuracy | Test accuracy | SHA-256 |
|---|---:|---:|---:|---|
| ConvNeXt V1-Atto | 5 | 80.20% | 79.47% | `211CAC6E7409A45A3A36F55904B2DC4D12F587B6CD199C3CACDF799EF88E2430` |
| ConvNeXt V2-Atto | 9 | 80.12% | 79.81% | `A09966B941F09405FA2E2E6BCF81728326B832ED071FAB19BEA882E66044D51A` |
| ConvNeXt V2-Atto without GRN | 17 | 75.30% | 74.84% | `73469927B8CD9081AE20EDFEF3EB620D1E9B20D2AFF88AC1D67FF437C0AC0B6D` |

Place the downloaded checkpoints at the paths expected by the evaluation scripts, or update those paths before running the scripts.
