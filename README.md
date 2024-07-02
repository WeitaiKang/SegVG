# SegVG-Transferring-Object-Bounding-Box-to-Segmentation-for-Visual-Grounding

**SiRi**: A Simple Selective Retraining Mechanism for Transformer-based Visual Grounding
========

This repository is an official PyTorch implementation of the ECCV 2022 paper [**SiRi: A Simple Selective Retraining Mechanism for Transformer-based Visual Grounding**](https://arxiv.org/abs/2207.13325).


# **Introduction**
we investigate a new training mechanism to improve the Transformer encoder, named **S**elect**i**ve **R**etra**i**ning (SiRi), which continually update the parameters of the encoder while periodically re-initialize the rest parameters as the training goes on. In this way, the model can be better optimized based on an enhanced encoder. Figure below shows the training process of SiRi. For more details. please refer to our paper.

![SiRi](.github/siri.png)

# **Updates**
   - **2022-7-25** Code and model link of SiRi in MDETR-like model on task REC
   - We will update the code and models on TransVG and other VL tasks such as RES.
# **Installation**
## Environment:
   - We provide instructions how to install dependencies via conda. First, clone the repository locally:
      ```
      git clone https://github.com/qumengxue/siri-vg.git
      ```
   - Make a new conda env and activate it:
      ```
      conda create -n siri python=3.8
      conda activate siri
      ```
   - Install the the packages in the requirements.txt: 
      ```
      pip install -r requirements.txt
      ```

## Dataset preparation
   - Prepare COCO training set ("train2014")
   - Download the [pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz) that are converted to coco format in MDETR. 
   - Modify the config file under `configs/` according to your dataset path, especially `coco_path`, `refexp_ann_path`.

For more installation details, please see the repository of [MDETR](https://github.com/ashkamath/mdetr), our code is built based on it.

## **Training**
   - For example, if with 2 decoders and 8 retraining periods in RefCOCOg, run
     ```
     sh refcocog.sh
     ```
   - For individual initial training, run
     ```
     python -m torch.distributed.launch   --nproc_per_node=4  --use_env main.py  --dataset_config configs/refcocog.json  --batch_size 18  --output-dir exps/refcocog_retrain_2decoder_1/   --ema   --lr 5e-5   --lr_backbone 5e-5   --text_encoder_lr 1e-5  --num_queries 16  --no_contrastive_align_loss  --cascade_num 2
     ```
## **Evaluation**
   - Training with running *.sh will automatically evaluate for each round of SiRi, so you can check it directly.
   - For individual model evaluation, run
      ```
      python -m torch.distributed.launch  --nproc_per_node=4  --use_env main.py  --dataset_config configs/refcocog.json  --batch_size 18 --output-dir exps/   --ema   --lr 5e-5   --lr_backbone 5e-5   --text_encoder_lr 1e-5  --num_queries 16  --no_contrastive_align_loss  --cascade_num 2  --resume exps/refcocog_1d.pth  --eval
      ```
## **Model Zoo**
**TASK1: Referring Expression Comprehension**
- **RefCOCO**

| Model             | val     | testA  | testB  | model |
|-------------------|---------|--------|--------|-------|
| MDETR\* +SiRi     | 85\.83  | 88\.56 | 81\.27 |[gdrive](https://drive.google.com/file/d/1nReXmFXbWhzpklsDX5BieoOXhYOGN1WY/view?usp=sharing)     |
| MDETR\* + MT SiRi | 85\.82  | 89\.11 | 81\.08 |[gdrive](https://drive.google.com/file/d/1LMvkQqoEMt_fRSOhaQf2zHMsaTE6mscF/view?usp=sharing)       |

- **RefCOCO+**

| Model             | val            | testA          | testB          | model |
|-------------------|----------------|----------------|----------------|-------|
| MDETR\* +SiRi     | 76\.68 (76.63) | 82\.01 (81.99) | 66\.33 (66.86) |[gdrive](https://drive.google.com/file/d/10XRIZXj4kZfhn5DprJ0clunhoa3xqXPF/view?usp=sharing)       |
| MDETR\* + MT SiRi | 77\.47 (77.53) | 83\.04 (82.47) | 67\.11 (67.89) |[gdrive](https://drive.google.com/file/d/1ItHWyHYogxcE3sBwrjLDeF8b24lfuBbQ/view?usp=sharing)       |

- **RefCOCOg**

| Model             | val      | test   | model |
|-------------------|----------|--------|-------|
| MDETR\* +SiRi     | 76\.63   | 76\.46 |[gdrive](https://drive.google.com/file/d/1m-FnDZ48F44xUvdpHnjzCkkn5VsPgm2v/view?usp=sharing)       |
| MDETR\* + MT SiRi | 77\.39   | 76\.80 |[gdrive](https://drive.google.com/file/d/1xEqdTnm5MQfabORr4X9lP9a9R1O8URRH/view?usp=sharing)       |

**TASK2: Referring Expression Segmentation**

Coming soon!

## **Citing SiRi**
```
@inproceedings{qu2022siri,
  title={SiRi: A Simple Selective Retraining Mechanism for Transformer-based Visual Grounding},
  author={Qu,Mengxue and Wu, Yu and Liu, Wu and Gong, Qiqi and Liang, Xiaodan and Olga, Russakovsky and Zhao, Yao and Wei, Yunchao},
  booktitle={ECCV},
  year={2022}
}
```
## **Acknowledgement**
Our code is built on the previous work [MDETR](https://github.com/ashkamath/mdetr).
