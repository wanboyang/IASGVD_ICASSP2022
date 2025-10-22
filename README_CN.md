# IASGVD: 基于信息注意力监督的接地视频描述

[![论文](https://img.shields.io/badge/论文-ICASSP_2022-blue)](https://ieeexplore.ieee.org/document/9746751)
[![许可证](https://img.shields.io/badge/许可证-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

> **语言**: [English](README.md) | [中文](README_CN.md)

---

## 摘要

本仓库实现了 **IASGVD (基于信息注意力监督的接地视频描述)**，这是在 ICASSP 2022 会议上提出的一种新颖框架，用于生成视频的自然语言描述，同时将描述定位到特定的视觉区域。我们的方法利用信息注意力监督，通过显式建模文本描述和视觉对象之间的关系来增强视频描述的质量和准确性。

## 🎯 主要特性

- **接地视频描述**: 生成描述的同时定位视频帧中的对象
- **信息注意力监督**: 增强注意力机制以实现更好的视觉-文本对齐
- **多模态集成**: 结合外观和运动特征以实现全面的视频理解
- **对象交互建模**: 使用transformer架构捕获对象之间的关系
- **弱监督学习**: 训练仅需要视频级标注

## 🏗️ 模型架构

该框架包含几个关键组件：

### 核心组件:
- **注意力模型 (AttModel)**: 视频描述生成的主要架构
- **Transformer编码器**: 处理视觉特征和对象交互
- **对象接地模块**: 定位描述中提到的对象
- **多尺度特征提取**: 结合区域和帧级特征

### 关键技术创新:
1. **信息注意力监督**: 引导注意力机制聚焦相关视觉区域
2. **对象交互建模**: 使用transformer捕获对象间关系
3. **多模态融合**: 集成RGB外观特征和运动特征
4. **弱监督**: 利用视频级标注进行训练

## 🚀 快速开始

### 安装

```bash
# 递归克隆仓库（包含子模块）
git clone --recursive https://github.com/wanboyang/IASGVD_ICASSP2022.git
cd IASGVD_ICASSP2022

# 创建环境
conda env create -f environment.yaml
conda activate anomaly_icme
```

### 快速设置

下载所有数据和预训练模型（总计216GB）：
```bash
bash tools/download_all.sh
```

### 入门代码

测试环境设置：
```bash
python main.py --batch_size 20 --cuda --checkpoint_path save/gvd_starter --id gvd_starter --language_eval
```

## 📊 数据准备

### 必需数据文件:
- **区域特征**: [下载](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois.tar.gz)
- **帧级特征**: [下载](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz)
- **标注文件**: [下载](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_prep.tar.gz)
- **描述文本**: [下载](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_captions.tar.gz)

### 数据集支持:
- **ActivityNet-Entities**: 主要的训练和评估数据集
- **隐藏测试集**: 可用于ANet-Entities对象定位挑战赛2020

## 🏋️ 训练

在 `cfgs/anet_res101_vg_feat_10x100prop_ip.yml` 中配置路径并运行：

```bash
CUDA_VISIBLE_DEVICES=1,0 python main.py --path_opt cfgs/anet_res101_vg_feat_10x100prop_ip.yml --batch_size 20 --cuda --checkpoint_path save/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --id topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --language_eval --w_att2 0.1 --w_grd 0 --w_cls 0.1 --obj_interact --overlap_type Both --att_model topdown --learning_rate 2e-4 --densecap_verbose --loss_type both3 --acc_num 4 --iou_thresh 0.5 --iop_thresh 0.9 --mGPUs | tee log/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4
```

## 🔍 推理和测试

### 验证集:
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --path_opt cfgs/anet_res101_vg_feat_10x100prop_ip.yml --batch_size 20 --cuda --num_workers 6 --max_epoch 50 --inference_only --start_from ./save/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --id topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --val_split validation --densecap_verbose --seq_length 20 --language_eval --obj_interact --eval_obj_grounding --grd_reference ./tools/anet_entities/data/anet_entities_cleaned_class_thresh50_test_skeleton.json --eval_obj_grounding_gt | tee log/eval-testing_split-topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4-beam1-standard-inference
```

**注意**: 评估需要至少9GB的可用GPU内存。

## 📈 性能表现

我们的方法在以下方面实现了最先进的性能：
- **ActivityNet-Entities**: 改进的描述质量和对象接地准确性
- **视频描述指标**: 更好的BLEU、METEOR和CIDEr分数
- **对象接地**: 增强的定位精度

## 📚 引用

如果您发现这项工作对您的研究有用，请引用：

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

## 🤝 致谢

我们感谢 [Grounded Video Description](https://github.com/facebookresearch/grounded-video-description) 和 [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities) 的贡献者提供的优秀框架和数据集。

## 📧 联系方式

如有问题和建议，请联系：
- **万博洋** - wanboyangjerry@163.com
