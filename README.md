# Road-detection-A-Vision-Transformer-Based-Study-and-fusion-comparison-
Camera‑LiDAR fusion for road detection using SegFormer‑B2 on KITTI‑360. 8‑fold CV shows RGB baseline outperforms fusion. Code to reproduce paper experiments, depth map generation, and visualization.

This repository contains the official implementation of the paper:

> Anthony Sigama, Baohua Guo, James Chakwizira, Tendayi Gondo, David Bassir, Sen Wang, Weifan Gu.  
> Camera-Lidar Fusion for Road Detection: A Vision Transformer-Based Study*  
> (Under review)

Visual abstract

<img src="visual_abstract.png" width="600" alt="Visual abstract: RGB baseline vs early/late/cross-attention fusion on KITTI-360">


Abstract
We evaluate camera‑LiDAR fusion for road segmentation using SegFormer‑B2 on KITTI‑360.  
Our experiments show that an RGB‑only baseline achieves the best Road IoU (0.9600) and lowest false positives, while fusion provides only marginal robustness improvements at twice the computational cost. We propose a **Fusion Justification Test** to guide multimodal decisions.

Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- albumentations, opencv‑python, numpy, tqdm, matplotlib, scipy

Install all dependencies with:


