# All results

Scores in %, forgetting in percentage points; mean ± standard deviation over seeds. Score: top-1 accuracy (classification) or mAP@[.5:.95] (detection). Second metric: macro-F1 or mAP@0.5. Forgetting: drop on ImageNetV2 top-1 (classification) or COCO val2017 mAP (detection).

## Main results

| Task | Model | Dataset | Data (%) | Method | LoRA rank | Score metric | Second metric | Forgetting measured on | Score (%) | Second metric (%) | Forgetting (pts) | Forgetting, BN stats restored (pts) | Trainable (%) | Peak memory (GB) | Time per epoch (s) | Training time per run (min) | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Classification | ResNet-50 | EuroSAT | 1 | Linear probe | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 87.14 ± 0.80 | 86.45 ± 0.73 | 0.00 ± 0.00 | – | 0.09 | 0.83 | – | 0.79 | 3 |
| Classification | ResNet-50 | EuroSAT | 1 | LoRA | 16 | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 88.55 ± 0.91 | 88.25 ± 0.89 | 39.96 ± 1.30 | – | 5.09 | 4.43 | 0.58 | 8.61 | 3 |
| Classification | ResNet-50 | EuroSAT | 1 | Full fine-tuning | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 87.89 ± 0.98 | 87.48 ± 1.06 | 63.35 ± 0.77 | 66.72 ± 1.86 | 100.00 | 3.30 | 0.53 | 5.62 | 3 |
| Classification | ResNet-50 | EuroSAT | 100 | Linear probe | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 96.04 ± 0.09 | 95.85 ± 0.09 | 0.01 ± 0.01 | – | 0.09 | 0.83 | – | 1.09 | 3 |
| Classification | ResNet-50 | EuroSAT | 100 | LoRA | 16 | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 98.43 ± 0.04 | 98.36 ± 0.04 | 52.94 ± 4.25 | – | 5.09 | 4.43 | 42.12 | 7.75 | 3 |
| Classification | ResNet-50 | EuroSAT | 100 | Full fine-tuning | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 98.44 ± 0.18 | 98.39 ± 0.18 | 67.43 ± 0.21 | 68.69 ± 0.06 | 100.00 | 3.30 | 32.51 | 5.94 | 3 |
| Classification | ResNet-50 | Flowers-102 | 100 | Linear probe | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 82.39 ± 0.28 | 81.45 ± 0.31 | -0.01 ± 0.01 | – | 0.88 | 0.83 | – | 0.64 | 3 |
| Classification | ResNet-50 | Flowers-102 | 100 | LoRA | 16 | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 89.54 ± 0.21 | 89.14 ± 0.14 | 36.35 ± 2.57 | – | 5.84 | 4.43 | 2.93 | 2.81 | 3 |
| Classification | ResNet-50 | Flowers-102 | 100 | Full fine-tuning | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 90.45 ± 0.80 | 90.06 ± 0.56 | 34.49 ± 0.46 | 63.90 ± 0.78 | 100.00 | 3.30 | 2.44 | 2.20 | 3 |
| Classification | ViT-B/16 | EuroSAT | 1 | Linear probe | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 88.17 ± 0.64 | 87.54 ± 0.72 | 0.00 ± 0.00 | – | 0.01 | 0.70 | – | 0.77 | 3 |
| Classification | ViT-B/16 | EuroSAT | 1 | LoRA | 16 | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 93.28 ± 0.73 | 93.01 ± 0.71 | 1.15 ± 0.22 | – | 1.04 | 4.25 | 0.64 | 10.46 | 3 |
| Classification | ViT-B/16 | EuroSAT | 1 | Full fine-tuning | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 92.70 ± 0.39 | 92.44 ± 0.44 | 2.25 ± 0.42 | – | 100.00 | 5.90 | 0.74 | 9.55 | 3 |
| Classification | ViT-B/16 | EuroSAT | 100 | Linear probe | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 97.02 ± 0.03 | 96.88 ± 0.04 | 0.03 ± 0.05 | – | 0.01 | 0.70 | – | 1.46 | 3 |
| Classification | ViT-B/16 | EuroSAT | 100 | LoRA | 16 | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 98.99 ± 0.06 | 98.94 ± 0.06 | 2.31 ± 0.26 | – | 1.04 | 4.25 | 52.24 | 9.56 | 3 |
| Classification | ViT-B/16 | EuroSAT | 100 | Full fine-tuning | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 98.98 ± 0.05 | 98.93 ± 0.05 | 3.52 ± 0.14 | – | 100.00 | 5.90 | 62.82 | 11.19 | 3 |
| Classification | ViT-B/16 | Flowers-102 | 100 | Linear probe | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 99.06 ± 0.04 | 98.91 ± 0.10 | 0.08 ± 0.00 | – | 0.09 | 0.70 | – | 0.63 | 3 |
| Classification | ViT-B/16 | Flowers-102 | 100 | LoRA | 16 | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 99.26 ± 0.17 | 99.14 ± 0.26 | 1.72 ± 0.14 | – | 1.12 | 4.25 | 3.18 | 2.77 | 3 |
| Classification | ViT-B/16 | Flowers-102 | 100 | Full fine-tuning | – | Top-1 accuracy | Macro-F1 | ImageNetV2 top-1 | 98.83 ± 0.20 | 98.77 ± 0.17 | 11.74 ± 0.91 | – | 100.00 | 5.90 | 3.76 | 3.01 | 3 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Head only | – | mAP@[.5:.95] | mAP@0.5 | COCO val2017 mAP@[.5:.95] | 68.94 ± 0.25 | 88.41 ± 0.32 | -0.00 ± 0.000403674 | – | 0.13 | 9.73 | 9.87 | 7.03 | 3 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | LoRA | 16 | mAP@[.5:.95] | mAP@0.5 | COCO val2017 mAP@[.5:.95] | 67.91 ± 0.64 | 87.38 ± 0.41 | 4.64 ± 0.60 | – | 2.58 | 10.56 | 14.61 | 10.10 | 3 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Full fine-tuning | – | mAP@[.5:.95] | mAP@0.5 | COCO val2017 mAP@[.5:.95] | 64.49 ± 0.40 | 85.27 ± 0.26 | 14.94 ± 0.55 | – | 100.00 | 11.57 | 15.28 | 10.31 | 3 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 100 | Head only | – | mAP@[.5:.95] | mAP@0.5 | COCO val2017 mAP@[.5:.95] | 69.77 ± 0.30 | 89.21 ± 0.35 | -0.01 ± 0.00 | – | 0.13 | 9.73 | 86.83 | 18.41 | 3 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 100 | LoRA | 16 | mAP@[.5:.95] | mAP@0.5 | COCO val2017 mAP@[.5:.95] | 71.14 ± 0.28 | 89.12 ± 0.03 | 7.98 ± 1.10 | – | 2.58 | 10.56 | 123.51 | 25.81 | 3 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 100 | Full fine-tuning | – | mAP@[.5:.95] | mAP@0.5 | COCO val2017 mAP@[.5:.95] | 70.14 ± 0.35 | 89.38 ± 0.36 | 15.32 ± 0.85 | – | 100.00 | 11.57 | 169.00 | 35.21 | 3 |

## Rank sweeps

| Task | Model | Dataset | Data (%) | LoRA rank | Score metric | Forgetting measured on | Score (%) | Forgetting (pts) | Trainable (%) | Peak memory (GB) | Time per epoch (s) | Training time per run (min) | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Classification | ViT-B/16 | Flowers-102 | 100 | 1 | Top-1 accuracy | ImageNetV2 top-1 | 99.20 ± 0.03 | 1.12 ± 0.27 | 0.16 | 4.23 | 3.11 | 2.59 | 3 |
| Classification | ViT-B/16 | Flowers-102 | 100 | 4 | Top-1 accuracy | ImageNetV2 top-1 | 99.10 ± 0.13 | 1.23 ± 0.13 | 0.35 | 4.23 | 3.16 | 2.63 | 3 |
| Classification | ViT-B/16 | Flowers-102 | 100 | 16 | Top-1 accuracy | ImageNetV2 top-1 | 99.25 ± 0.13 | 1.77 ± 0.33 | 1.12 | 4.25 | 3.18 | 2.65 | 3 |
| Classification | ViT-B/16 | Flowers-102 | 100 | 64 | Top-1 accuracy | ImageNetV2 top-1 | 99.04 ± 0.11 | 2.67 ± 0.26 | 4.21 | 4.32 | 3.16 | 2.63 | 3 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | 1 | mAP@[.5:.95] | COCO val2017 mAP@[.5:.95] | 69.02 ± 0.39 | 2.41 ± 0.12 | 0.28 | 10.56 | 13.16 | 9.06 | 2 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | 4 | mAP@[.5:.95] | COCO val2017 mAP@[.5:.95] | 68.65 ± 0.04 | 4.20 ± 0.81 | 0.74 | 10.56 | 13.96 | 9.66 | 2 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | 16 | mAP@[.5:.95] | COCO val2017 mAP@[.5:.95] | 67.86 ± 0.90 | 4.29 ± 0.16 | 2.58 | 10.56 | 15.04 | 10.42 | 2 |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | 64 | mAP@[.5:.95] | COCO val2017 mAP@[.5:.95] | 64.46 ± 0.13 | 12.66 ± 0.11 | 9.95 | 10.57 | 13.53 | 9.33 | 2 |

## LR tuning

| Task | Model | Dataset | Data (%) | Method | Learning rate | Validation score (%) | Test score (%) | Forgetting (pts) | Selected | Time per epoch (s) | Training time per run (min) | Score metric |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Classification | ResNet-50 | EuroSAT | 1 | Linear probe | 0.001 | 84.04 | 83.83 | 0.02 |  | – | 1.08 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | Linear probe | 0.003 | 86.52 | 86.56 | -0.02 |  | – | 0.75 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | Linear probe | 0.01 | 86.74 | 86.15 | 0.02 | yes | – | 0.70 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | Linear probe | 0.03 | 86.48 | 85.85 | 0.02 |  | – | 0.71 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | LoRA | 0.0003 | 87.67 | 86.63 | 46.13 |  | 0.64 | 8.68 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | LoRA | 0.001 | 89.59 | 88.28 | 32.27 | yes | 0.67 | 9.07 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | LoRA | 0.003 | 89.22 | 89.06 | 39.24 |  | 0.60 | 8.40 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | LoRA | 0.01 | 73.00 | 72.96 | 38.31 |  | 0.55 | 8.11 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | Full fine-tuning | 3e-05 | 52.52 | 52.52 | 62.73 |  | 0.62 | 6.85 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | Full fine-tuning | 0.0001 | 73.15 | 75.46 | 63.65 |  | 0.56 | 5.74 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | Full fine-tuning | 0.0003 | 87.52 | 87.69 | 58.59 |  | 0.53 | 5.66 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | Full fine-tuning | 0.001 | 88.30 | 88.20 | 65.89 | yes | 0.52 | 5.55 | Top-1 accuracy |
| Classification | ResNet-50 | EuroSAT | 1 | Full fine-tuning | 0.003 | 87.04 | 86.48 | 67.93 |  | 0.52 | 5.71 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Linear probe | 0.001 | 81.76 | 79.66 | 0.00 |  | – | 0.54 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Linear probe | 0.003 | 84.12 | 81.41 | -0.01 |  | – | 0.60 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Linear probe | 0.01 | 84.31 | 82.71 | -0.01 | yes | – | 0.60 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Linear probe | 0.03 | 84.31 | 82.40 | -0.02 |  | – | 0.76 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | LoRA | 0.0003 | 89.22 | 84.75 | 32.30 |  | 3.02 | 3.37 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | LoRA | 0.001 | 91.27 | 87.56 | 22.38 |  | 2.80 | 3.11 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | LoRA | 0.003 | 91.67 | 88.60 | 18.74 |  | 2.86 | 2.50 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | LoRA | 0.01 | 91.86 | 90.67 | 39.39 | yes | 2.83 | 2.58 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Full fine-tuning | 3e-05 | 16.18 | 15.73 | 14.82 |  | 2.51 | 2.24 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Full fine-tuning | 0.0001 | 65.29 | 63.46 | 17.48 |  | 2.44 | 2.19 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Full fine-tuning | 0.0003 | 88.43 | 85.15 | 28.37 |  | 2.42 | 2.19 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Full fine-tuning | 0.001 | 91.86 | 89.33 | 35.12 | yes | 2.29 | 2.07 | Top-1 accuracy |
| Classification | ResNet-50 | Flowers-102 | 100 | Full fine-tuning | 0.003 | 91.47 | 89.87 | 56.62 |  | 2.17 | 1.92 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | Linear probe | 0.001 | 87.41 | 88.31 | 0.00 |  | – | 0.99 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | Linear probe | 0.003 | 88.26 | 88.91 | 0.00 | yes | – | 0.75 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | Linear probe | 0.01 | 87.56 | 88.48 | 0.00 |  | – | 0.73 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | LoRA | 0.0003 | 93.37 | 93.52 | 1.07 | yes | 0.64 | 10.55 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | LoRA | 0.001 | 93.07 | 93.54 | 1.39 |  | 0.64 | 10.60 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | LoRA | 0.003 | 92.78 | 92.52 | 8.33 |  | 0.63 | 10.45 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | Full fine-tuning | 3e-06 | 91.30 | 91.04 | 1.10 |  | 0.91 | 10.63 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | Full fine-tuning | 1e-05 | 93.89 | 92.80 | 1.77 | yes | 0.73 | 9.47 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | Full fine-tuning | 3e-05 | 92.93 | 91.70 | 3.46 |  | 0.70 | 9.37 | Top-1 accuracy |
| Classification | ViT-B/16 | EuroSAT | 1 | Full fine-tuning | 0.0001 | 92.44 | 91.74 | 10.40 |  | 0.70 | 9.14 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | Linear probe | 0.001 | 98.63 | 98.98 | 0.08 |  | – | 0.64 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | Linear probe | 0.003 | 98.82 | 99.11 | 0.08 | yes | – | 0.69 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | Linear probe | 0.01 | 98.63 | 98.85 | 0.08 |  | – | 0.65 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | LoRA | 0.0003 | 99.31 | 99.15 | 1.40 | yes | 3.60 | 2.94 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | LoRA | 0.001 | 99.31 | 99.07 | 2.38 |  | 3.12 | 2.61 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | LoRA | 0.003 | 98.43 | 98.05 | 11.08 |  | 3.27 | 2.85 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | Full fine-tuning | 3e-06 | 90.49 | 88.73 | 5.84 |  | 3.50 | 2.59 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | Full fine-tuning | 1e-05 | 97.75 | 97.45 | 9.26 |  | 3.82 | 3.02 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | Full fine-tuning | 3e-05 | 99.12 | 98.62 | 10.70 | yes | 3.79 | 3.02 | Top-1 accuracy |
| Classification | ViT-B/16 | Flowers-102 | 100 | Full fine-tuning | 0.0001 | 98.53 | 98.06 | 11.49 |  | 3.78 | 3.03 | Top-1 accuracy |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Head only | 0.0003 | 68.65 | 69.17 | -0.00 |  | 5.44 | 3.98 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Head only | 0.001 | 68.69 | 69.13 | -0.00 |  | 6.70 | 4.85 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Head only | 0.003 | 68.90 | 69.18 | -0.00 | yes | 6.36 | 4.60 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Head only | 0.01 | 67.63 | 67.23 | -0.00 |  | 9.21 | 6.61 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | LoRA | 0.0001 | 66.15 | 67.07 | 7.60 |  | 8.65 | 6.03 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | LoRA | 0.0003 | 66.12 | 67.50 | 7.40 |  | 9.16 | 6.35 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | LoRA | 0.001 | 66.98 | 67.02 | 7.72 | yes | 9.70 | 6.79 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | LoRA | 0.003 | 64.70 | 64.03 | 13.04 |  | 13.10 | 9.05 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Full fine-tuning | 3e-05 | 62.61 | 63.87 | 14.48 |  | 16.19 | 10.87 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Full fine-tuning | 0.0001 | 64.16 | 64.49 | 14.58 | yes | 15.75 | 10.65 | mAP@[.5:.95] |
| Detection | RT-DETR-R18 | Pascal VOC 2007 | 10 | Full fine-tuning | 0.0003 | 60.40 | 58.63 | 20.38 |  | 15.08 | 10.17 | mAP@[.5:.95] |

## BatchNorm check

| Model | Dataset | Data (%) | Learning rate | Seed | Test accuracy (%) | Forgetting (pts) | Forgetting, BN stats restored (pts) | Share caused by BN statistics (%) |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 | EuroSAT | 1 | 3e-05 | 0 | 52.44 | 64.41 | 0.93 | 98.56 |
| ResNet-50 | EuroSAT | 1 | 3e-05 | 1 | 51.30 | 62.38 | 0.59 | 99.05 |
| ResNet-50 | EuroSAT | 1 | 3e-05 | 2 | 53.43 | 61.01 | 0.97 | 98.41 |
|  |  | – | nan | mean | 52.39 | 62.60 | 0.83 | 98.67 |
|  |  | – | nan | std | 1.07 | 1.71 | 0.21 | 0.34 |
