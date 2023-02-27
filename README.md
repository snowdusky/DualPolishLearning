# Mind the Gap: Polishing Pseudo labels for Accurate Semi-supervised Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mind-the-gap-polishing-pseudo-labels-for/semi-supervised-object-detection-on-coco-1)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-1?p=mind-the-gap-polishing-pseudo-labels-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mind-the-gap-polishing-pseudo-labels-for/semi-supervised-object-detection-on-coco-5)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-5?p=mind-the-gap-polishing-pseudo-labels-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mind-the-gap-polishing-pseudo-labels-for/semi-supervised-object-detection-on-coco-10)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-10?p=mind-the-gap-polishing-pseudo-labels-for)



[Lei Zhang*](https://scholar.google.com/citations?user=0Kg6Gi4AAAAJ&hl=zh-CN), [Yuxuan Sun*](https://scholar.google.com/citations?user=HQsBJ6IAAAAJ&hl=zh-CN), Wei Wei
![](./teaser/framework.png)




This repo is the official implementation of our AAAI2023 paper ["Mind the Gap: Polishing Pseudo labels for Accurate Semi-supervised Object Detection"](https://arxiv.org/abs/2207.08185) on PyTorch.


## Results

### COCO


#### 1% labeled data
| Method | mAP| Model Weights |Config Files|
| :----: | -------| ----- |----|
| Baseline|  10.0 |-|[Config](configs/coco/coco_base.py)|
| Ours    | 23.80 |-|[Config](configs/coco/coco_part.py)|


#### 5% labeled data
| Method | mAP| Model Weights |Config Files|
| :----: | -------| ----- |----|
| Baseline| 10.00 |-|[Config](configs/coco/coco_base.py)|
| Ours    | 32.15 |-|[Config](configs/coco/coco_part.py)|

#### 10% labeled data
| Method | mAP| Model Weights |Config Files|
| :----: | -------| ----- |----|
| Baseline| 20.92 |-|[Config](configs/coco/coco_base.py)|
| Ours    | 35.30 |-|[Config](configs/coco/coco_part.py)|


### VOC

| Method | mAP | AP50 | Model Weights |Config Files|
| :----: | -------| ----- |----|----|
| Baseline| 43.00 | 76.70 |-|[Config](configs/voc/voc07_base.py)|
| Ours    | 52.40 | 82.50 |-|[Config](configs/voc/voc07_rf12.py)|


## Usage

Since this repo is built on the [Soft Teacher](https://github.com/microsoft/SoftTeacher), some setup instructions are cloned from it.

### Requirements
- `Ubuntu 16.04`
- `Anaconda3` with `python=3.6`
- `Pytorch=1.9.0`
- `mmdetection=2.16.0+fe46ffe`
- `mmcv=1.3.9`

### Installation
```
pip install -r requirements.txt
cd thirdparty/mmdetection && pip install -e .
cd ../.. && pip install -e .
```

### Data Preparation

- Download the COCO dataset
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA should be a directory contains coco dataset.
# For eg.:
# YOUR_DATA/
#  coco/
#     train2017/
#     val2017/
#     unlabeled2017/
#     annotations/
ln -s ${YOUR_DATA} data
bash tools/dataset/prepare_coco_data.sh conduct
```
For concrete instructions of what should be downloaded, please refer to `tools/dataset/prepare_coco_data.sh` line [`11-24`](https://github.com/microsoft/SoftTeacher/blob/863d90a3aa98615be3d156e7d305a22c2a5075f5/tools/dataset/prepare_coco_data.sh#L11)


### Training
- To train model on the **partial labeled data** setting:
```shell script
tools/dist_train.sh ./configs/coco/coco_part.py 8 --work-dir=<YOUR_WORKDIR_PATH> --cfg-options fold=<FOLD> percent=<PERCENT_LABELED_DATA> 
```
For example, we could run the following scripts to train our model on 10% labeled data of fold 1 with 8 GPUs:
```shell script
tools/dist_train.sh ./configs/coco/coco_part.py 8 --work-dir=<YOUR_WORKDIR_PATH> --cfg-options fold=1 percent=10 
```
- To train model under VOC07 (as labeled set) and VOC12 (as unlabeled set):
```shell script
tools/dist_train.sh ./configs/voc/voc07_rf12.py 8 --work-dir=<YOUR_WORKDIR_PATH>
```
All experiments are trained on 8 GPUs by default.



### Evaluation
```
bash tools/dist_test.sh <CONFIG_FILE_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval bbox --cfg-options model.test_cfg.rcnn.score_thr=<THR>
```

## Acknowledgement

Polishing Teacher builds on previous works code base such as [Soft Teacher](https://github.com/microsoft/SoftTeacher) and [mmdetection](https://github.com/open-mmlab/mmdetection). Thanks for their wonderful works.

## Citation
If you found this code helpful, feel free to cite our work:
```bib
@article{zhang2022mind,
  title={Mind the Gap: Polishing Pseudo labels for Accurate Semi-supervised Object Detection},
  author={Zhang, Lei and Sun, Yuxuan and Wei, Wei},
  journal={arXiv preprint arXiv:2207.08185},
  year={2022}
}
```
