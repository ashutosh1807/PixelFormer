[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention-attention-everywhere-monocular/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=attention-attention-everywhere-monocular)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention-attention-everywhere-monocular/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=attention-attention-everywhere-monocular)

## PixelFormer: Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention
This is the official PyTorch implementation for WACV 2023 paper 'Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention'.

**[Paper](https://arxiv.org/pdf/2210.09071)** <br />


### Installation
```
conda create -n pixelformer python=3.8
conda activate pixelformer
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1
pip install matplotlib, tqdm, tensorboardX, timm, mmcv
```


### Datasets
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/cleinc/bts), and then modify the data path in the config files to your dataset locations.


### Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the NYUv2 model:
```
python pixelformer/train.py configs/arguments_train_nyu.txt
```

Training the KITTI model:
```
python pixelformer/train.py configs/arguments_train_kittieigen.txt
```


### Evaluation
Evaluate the NYUv2 model:
```
python pixelformer/eval.py configs/arguments_eval_nyu.txt
```

Evaluate the KITTI model:
```
python pixelformer/eval.py configs/arguments_eval_kittieigen.txt
```

## Pretrained Models
* You can download the pretrained models "nyu.pt" and "kitti.pt" from [here](https://drive.google.com/drive/folders/1Feo67jEbccqa-HojTHG7ljTXOW2yuX-X?usp=share_link).



### Acknowledgements
Most of the code has been adpated from CVPR 2022 paper [NewCRFS](https://github.com/aliyun/NeWCRFs). We thank Weihao Yuan for releasing the source code for the same.

Also, thanks to Microsoft Research Asia for opening source of the excellent work [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
