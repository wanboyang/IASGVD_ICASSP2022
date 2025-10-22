# IASGVD: Informative Attention Supervision for Grounded Video Description

[![Paper](https://img.shields.io/badge/Paper-ICASSP_2022-blue)](https://ieeexplore.ieee.org/document/9746751)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

> **Language**: [English](README.md) | [中文](README_CN.md)

---

## Abstract

This repository implements **IASGVD (Informative Attention Supervision for Grounded Video Description)**, a novel framework presented at ICASSP 2022 for generating natural language descriptions of videos while simultaneously grounding the descriptions to specific visual regions. Our approach leverages informative attention supervision to enhance the quality and accuracy of video descriptions by explicitly modeling the relationships between textual descriptions and visual objects.

## 🎯 Key Features

- **Grounded Video Description**: Generates descriptions while localizing objects in video frames
- **Informative Attention Supervision**: Enhances attention mechanisms for better visual-textual alignment
- **Multi-modal Integration**: Combines appearance and motion features for comprehensive video understanding
- **Object Interaction Modeling**: Captures relationships between objects using transformer architectures
- **Weakly Supervised Learning**: Requires only video-level annotations for training

## 🏗️ Model Architecture

The framework consists of several key components:

### Core Components:
- **Attention Model (AttModel)**: Main architecture for video description generation
- **Transformer Encoder**: Processes visual features and object interactions
- **Object Grounding Module**: Localizes objects mentioned in descriptions
- **Multi-scale Feature Extraction**: Combines region and frame-level features

### Key Technical Innovations:
1. **Informative Attention Supervision**: Guides attention mechanisms to focus on relevant visual regions
2. **Object Interaction Modeling**: Uses transformers to capture relationships between objects
3. **Multi-modal Fusion**: Integrates RGB appearance features and motion features
4. **Weak Supervision**: Leverages video-level annotations for training

## 🚀 Quick Start

### Installation

```bash
# Clone repository recursively (includes submodules)
git clone --recursive https://github.com/wanboyang/IASGVD_ICASSP2022.git
cd IASGVD_ICASSP2022

# Create environment
conda env create -f environment.yaml
conda activate anomaly_icme
```

### Quick Setup

Download all data and pre-trained models (total 216GB):
```bash
bash tools/download_all.sh
```

### Starter Code

Test your environment setup:
```bash
python main.py --batch_size 20 --cuda --checkpoint_path save/gvd_starter --id gvd_starter --language_eval
```

## 📊 Data Preparation

### Required Data Files:
- **Region Features**: [Download](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois.tar.gz)
- **Frame-wise Features**: [Download](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz)
- **Annotations**: [Download](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_prep.tar.gz)
- **Captions**: [Download](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_captions.tar.gz)

### Dataset Support:
- **ActivityNet-Entities**: Primary dataset for training and evaluation
- **Hidden Test Set**: Available for ANet-Entities Object Localization Challenge 2020

## 🏋️ Training

Configure paths in `cfgs/anet_res101_vg_feat_10x100prop_ip.yml` and run:

```bash
CUDA_VISIBLE_DEVICES=1,0 python main.py --path_opt cfgs/anet_res101_vg_feat_10x100prop_ip.yml --batch_size 20 --cuda --checkpoint_path save/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --id topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --language_eval --w_att2 0.1 --w_grd 0 --w_cls 0.1 --obj_interact --overlap_type Both --att_model topdown --learning_rate 2e-4 --densecap_verbose --loss_type both3 --acc_num 4 --iou_thresh 0.5 --iop_thresh 0.9 --mGPUs | tee log/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4
```

## 🔍 Inference and Testing

### Validation Split:
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --path_opt cfgs/anet_res101_vg_feat_10x100prop_ip.yml --batch_size 20 --cuda --num_workers 6 --max_epoch 50 --inference_only --start_from ./save/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --id topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --val_split validation --densecap_verbose --seq_length 20 --language_eval --obj_interact --eval_obj_grounding --grd_reference ./tools/anet_entities/data/anet_entities_cleaned_class_thresh50_test_skeleton.json --eval_obj_grounding_gt | tee log/eval-testing_split-topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4-beam1-standard-inference
```

**Note**: Requires at least 9GB of free GPU memory for evaluation.

## 📈 Performance

Our method achieves state-of-the-art performance on:
- **ActivityNet-Entities**: Improved description quality and object grounding accuracy
- **Video Description Metrics**: Better BLEU, METEOR, and CIDEr scores
- **Object Grounding**: Enhanced localization precision

## 📚 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{wan2022informative,
  title={Informative Attention Supervision for Grounded Video Description},
  author={Wan, Boyang and Jiang, Wenhui and Fang, Yuming},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1955--1959},
  year={2022},
  organization={IEEE}
}
```

## 🤝 Acknowledgements

We thank the contributors of [Grounded Video Description](https://github.com/facebookresearch/grounded-video-description) and [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities) for their excellent frameworks and datasets.

## 📧 Contact

For questions and suggestions, please contact:
- **Boyang Wan** - wanboyangjerry@163.com
