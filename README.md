# Informative Attention Supervision for Grounded Video Description


This repo hosts the source code for our paper [Informative Attention Supervision for Grounded Video Description](https://ieeexplore.ieee.org/document/9746751). It supports [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities) dataset.

## Quick Start
### Preparations
Follow the instructions 1 to 3 in the [Requirements](#req) section to install required packages.

### Download everything
Simply run the following command to download all the data and pre-trained models (total 216GB):
```
bash tools/download_all.sh
```

### Starter code
Run the following eval code to test if your environment is setup:

```
python main.py --batch_size 20 --cuda --checkpoint_path save/gvd_starter --id gvd_starter --language_eval
```
You can now skip to the [Training and Validation](#train) section!


## <a name='req'></a> Requirements (Recommended)
1) Clone the repo recursively:

Make sure all the submodules [densevid_eval](https://github.com/LuoweiZhou/densevid_eval_spice) and [coco-caption](https://github.com/tylin/coco-caption) are included.

2) Rebuilding the environment via Anaconda.

```
conda env create -f environment.yaml
```


3 (Optional) If you choose to not use `download_all.sh`, be sure to install JAVA and download Stanford CoreNLP for SPICE (see [here](https://github.com/tylin/coco-caption)). Also, download and place the reference [file](https://github.com/jiasenlu/coco-caption/blob/master/annotations/caption_flickr30k.json) under `coco-caption/annotations`. Download [Stanford CoreNLP 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html) for grounding evaluation and place the uncompressed folder under the `tools` directory.


## Data Preparation
Updates on 04/15/2020: Feature files for the **hidden** test set, used in ANet-Entities Object Localization Challenge 2020, are available to download ([region features](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois_hidden_test.tar.gz) and [frame-wise features](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d_hidden_test.tar.gz)). Make sure you move the additional *.npy files over to your folder `fc6_feat_100rois` and `rgb_motion_1d`, respectively. The following files have been updated to include the **hidden** test set or video IDs: `anet_detection_vg_fc6_feat_100rois.h5`, `anet_entities_prep.tar.gz`, and `anet_entities_captions.tar.gz`.

Download the preprocessed annotation files from [here](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_prep.tar.gz), uncompress and place them under `data/anet`. Or you can reproduce them all using the data from ActivityNet-Entities [repo](https://github.com/facebookresearch/ActivityNet-Entities) and the preprocessing script `prepro_dic_anet.py` under `prepro`. Then, download the ground-truth caption annotations (under our val/test splits) from [here](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_captions.tar.gz) and same place under `data/anet`.

The region features and detections are available for download ([feature](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois.tar.gz) and [detection](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_detection_vg_fc6_feat_100rois.h5)). The region feature file should be decompressed and placed under your feature directory. We refer to the region feature directory as `feature_root` in the code. The H5 region detection (proposal) file is referred to as `proposal_h5` in the code. To extract feature for customized dataset (or brave folks for ANet-Entities as well), refer to the feature extraction tool [here](https://github.com/LuoweiZhou/detectron-vlp).

The frame-wise appearance (with suffix `_resnet.npy`) and motion (with suffix `_bn.npy`) feature files are available [here](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz). We refer to this directory as `seg_feature_root`.

Other auxiliary files, such as the weights from Detectron fc7 layer, are available [here](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz). Uncompress and place under the `data` directory.


## <a name='train'></a> Training and Validation
Modify the config file `cfgs/anet_res101_vg_feat_10x100prop_ip.yml` with the correct dataset and feature paths (or through symlinks). Link `tools/anet_entities` to your ANet-Entities dataset root location. Create new directories `log` and `results` under the root directory to save log and result files.

```
CUDA_VISIBLE_DEVICES=1,0 python main.py --path_opt cfgs/anet_res101_vg_feat_10x100prop_ip.yml  --batch_size 20 --cuda --checkpoint_path save/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --id topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --language_eval --w_att2 0.1 --w_grd 0 --w_cls 0.1 --obj_interact --overlap_type Both --att_model topdown --learning_rate 2e-4 --densecap_verbose --loss_type both3 --acc_num 4 --iou_thresh 0.5 --iop_thresh 0.9 --mGPUs | tee log/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4
```

(Optional) Remove `--mGPUs` to run in single-GPU mode.




## Inference and Testing
For supervised models (`ID=topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4`):

```
CUDA_VISIBLE_DEVICES=1 python main.py --path_opt  cfgs/anet_res101_vg_feat_10x100prop_ip.yml --batch_size 20 --cuda --num_workers 6 --max_epoch 50 --inference_only --start_from ./save/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --id topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --val_split validation  --densecap_verbose --seq_length 20 --language_eval --obj_interact --eval_obj_grounding  --grd_reference ./tools/anet_entities/data/anet_entities_cleaned_class_thresh50_test_skeleton.json --eval_obj_grounding_gt| tee log/eval-testing_split-topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4-beam1-standard-inference
```
```
CUDA_VISIBLE_DEVICES=1 python main.py --path_opt  cfgs/anet_res101_vg_feat_10x100prop_ip.yml --batch_size 20 --cuda --num_workers 6 --max_epoch 50 --inference_only --start_from ./save/topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --id topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4 --val_split testing  --densecap_verbose --seq_length 20 --language_eval --obj_interact --eval_obj_grounding  --grd_reference ./tools/anet_entities/data/anet_entities_cleaned_class_thresh50_test_skeleton.json --eval_obj_grounding_gt| tee log/eval-testing_split-topdown_iou_iop_cls_attn_both3loss_w_att2_0.1_cuda11_accnum2e4-beam1-standard-inference
```

Arguments: `dc_references='./data/anet/anet_entities_val_1.json ./data/anet/anet_entities_val_2.json'`, `grd_reference='tools/anet_entities/data/anet_entities_cleaned_class_thresh50_trainval.json'` `val_split='validation'`.

You need at least 9GB of free GPU memory for the evaluation.


## Reference
Please acknowledge the following paper if you use the code:

```
@inproceedings{wan2022informative,
  title={Informative Attention Supervision for Grounded Video Description},
  author={Wan, Boyang and Jiang, Wenhui and Fang, Yuming},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1955--1959},
  year={2022},
  organization={IEEE}
}
```


## Acknowledgement
We thank project [Grounded Video Description](https://github.com/facebookresearch/grounded-video-description). 


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree. 

Portions of the source code are based on the [Grounded Video Description](https://github.com/facebookresearch/grounded-video-description).
