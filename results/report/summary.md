## Classification

| Stage | Model | Dataset | Data | Method | Rank | Test acc (%) | Macro-F1 (%) | Forgetting (pts) | Forgetting, BN stats restored (pts) | Trainable (%) | Peak mem (GB) | Time/epoch (s) | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | resnet50 | eurosat | 1% | Linear probe | – | 87.1 ± 0.8 | 86.4 ± 0.7 | 0.00 ± 0.00 | – | 0.09 | 0.83 | – | 3 |
| main | resnet50 | eurosat | 1% | LoRA | 16 | 88.5 ± 0.9 | 88.3 ± 0.9 | 39.96 ± 1.30 | – | 5.09 | 4.43 | 0.6 | 3 |
| main | resnet50 | eurosat | 1% | Full fine-tuning | – | 87.9 ± 1.0 | 87.5 ± 1.1 | 63.35 ± 0.77 | 66.72 ± 1.86 | 100.00 | 3.30 | 0.5 | 3 |
| main | resnet50 | eurosat | 100% | Linear probe | – | 96.0 ± 0.1 | 95.8 ± 0.1 | 0.01 ± 0.01 | – | 0.09 | 0.83 | – | 3 |
| main | resnet50 | eurosat | 100% | LoRA | 16 | 98.4 ± 0.0 | 98.4 ± 0.0 | 52.94 ± 4.25 | – | 5.09 | 4.43 | 42.1 | 3 |
| main | resnet50 | eurosat | 100% | Full fine-tuning | – | 98.4 ± 0.2 | 98.4 ± 0.2 | 67.43 ± 0.21 | 68.69 ± 0.06 | 100.00 | 3.30 | 32.5 | 3 |
| main | resnet50 | flowers102 | 100% | Linear probe | – | 82.4 ± 0.3 | 81.4 ± 0.3 | -0.01 ± 0.01 | – | 0.88 | 0.83 | – | 3 |
| main | resnet50 | flowers102 | 100% | LoRA | 16 | 89.5 ± 0.2 | 89.1 ± 0.1 | 36.35 ± 2.57 | – | 5.84 | 4.43 | 2.9 | 3 |
| main | resnet50 | flowers102 | 100% | Full fine-tuning | – | 90.5 ± 0.8 | 90.1 ± 0.6 | 34.49 ± 0.46 | 63.90 ± 0.78 | 100.00 | 3.30 | 2.4 | 3 |
| main | vit_base_patch16_224 | eurosat | 1% | Linear probe | – | 88.2 ± 0.6 | 87.5 ± 0.7 | 0.00 ± 0.00 | – | 0.01 | 0.70 | – | 3 |
| main | vit_base_patch16_224 | eurosat | 1% | LoRA | 16 | 93.3 ± 0.7 | 93.0 ± 0.7 | 1.15 ± 0.22 | – | 1.04 | 4.25 | 0.6 | 3 |
| main | vit_base_patch16_224 | eurosat | 1% | Full fine-tuning | – | 92.7 ± 0.4 | 92.4 ± 0.4 | 2.25 ± 0.42 | – | 100.00 | 5.90 | 0.7 | 3 |
| main | vit_base_patch16_224 | eurosat | 100% | Linear probe | – | 97.0 ± 0.0 | 96.9 ± 0.0 | 0.03 ± 0.05 | – | 0.01 | 0.70 | – | 3 |
| main | vit_base_patch16_224 | eurosat | 100% | LoRA | 16 | 99.0 ± 0.1 | 98.9 ± 0.1 | 2.31 ± 0.26 | – | 1.04 | 4.25 | 52.2 | 3 |
| main | vit_base_patch16_224 | eurosat | 100% | Full fine-tuning | – | 99.0 ± 0.1 | 98.9 ± 0.1 | 3.52 ± 0.14 | – | 100.00 | 5.90 | 62.8 | 3 |
| main | vit_base_patch16_224 | flowers102 | 100% | Linear probe | – | 99.1 ± 0.0 | 98.9 ± 0.1 | 0.08 ± 0.00 | – | 0.09 | 0.70 | – | 3 |
| main | vit_base_patch16_224 | flowers102 | 100% | LoRA | 16 | 99.3 ± 0.2 | 99.1 ± 0.3 | 1.72 ± 0.14 | – | 1.12 | 4.25 | 3.2 | 3 |
| main | vit_base_patch16_224 | flowers102 | 100% | Full fine-tuning | – | 98.8 ± 0.2 | 98.8 ± 0.2 | 11.74 ± 0.91 | – | 100.00 | 5.90 | 3.8 | 3 |
| rank | vit_base_patch16_224 | flowers102 | 100% | LoRA | 1 | 99.2 ± 0.0 | 99.1 ± 0.1 | 1.12 ± 0.27 | – | 0.16 | 4.23 | 3.1 | 3 |
| rank | vit_base_patch16_224 | flowers102 | 100% | LoRA | 4 | 99.1 ± 0.1 | 99.0 ± 0.1 | 1.23 ± 0.13 | – | 0.35 | 4.23 | 3.2 | 3 |
| rank | vit_base_patch16_224 | flowers102 | 100% | LoRA | 16 | 99.2 ± 0.1 | 99.1 ± 0.2 | 1.77 ± 0.33 | – | 1.12 | 4.25 | 3.2 | 3 |
| rank | vit_base_patch16_224 | flowers102 | 100% | LoRA | 64 | 99.0 ± 0.1 | 98.9 ± 0.2 | 2.67 ± 0.26 | – | 4.21 | 4.32 | 3.2 | 3 |

## Detection

| Model | Data | Method | VOC mAP (%) | VOC mAP@0.5 (%) | COCO forgetting (mAP pts) | Trainable (%) | Peak mem (GB) | Time/epoch (s) | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| PekingU/rtdetr_r18vd | 10% | Head only | 68.9 ± 0.2 | 88.4 ± 0.3 | -0.00 ± 0.00 | 0.13 | 9.73 | 9.9 | 3 |
| PekingU/rtdetr_r18vd | 10% | LoRA | 67.9 ± 0.6 | 87.4 ± 0.4 | 4.64 ± 0.60 | 2.58 | 10.56 | 14.6 | 3 |
| PekingU/rtdetr_r18vd | 10% | Full fine-tuning | 64.5 ± 0.4 | 85.3 ± 0.3 | 14.94 ± 0.55 | 100.00 | 11.57 | 15.3 | 3 |
| PekingU/rtdetr_r18vd | 100% | Head only | 69.8 ± 0.3 | 89.2 ± 0.3 | -0.01 ± 0.00 | 0.13 | 9.73 | 86.8 | 3 |
| PekingU/rtdetr_r18vd | 100% | LoRA | 71.1 ± 0.3 | 89.1 ± 0.0 | 7.98 ± 1.10 | 2.58 | 10.56 | 123.5 | 3 |
| PekingU/rtdetr_r18vd | 100% | Full fine-tuning | 70.1 ± 0.4 | 89.4 ± 0.4 | 15.32 ± 0.85 | 100.00 | 11.57 | 169.0 | 3 |
