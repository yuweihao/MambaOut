# [MambaOut: Do We Really Need Mamba for Vision?](https://arxiv.org/abs/2405.07992)

<p align="left">
<a href="https://arxiv.org/abs/2405.07992" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2405.07992-b31b1b.svg?style=flat" /></a>
<a href="https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

<p align="center"><em>In memory of Kobe Bryant</em></p>

> "What can I say, Mamba out." — *Kobe Bryant, NBA farewell speech, 2016*

<p align="center">
<img src="https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mamba_out.png" width="400"> <br>
<small>Image credit: https://www.ebay.ca/itm/264973452480</small>
</p>


This is a PyTorch implementation of MambaOut proposed by our paper "[MambaOut: Do We Really Need Mamba for Vision?](https://arxiv.org/abs/2303.16900)". 


![MambaOut first figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_first_figure.png)


## Requirements
PyTorch and timm 0.6.11 (`pip install timm==0.6.11`).

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## Models
### MambaOut trained on ImageNet
| Model | Resolution | Params | MACs | Top1 Acc |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |
| [mambaout_femto](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth) | 224 | 7.3M | 1.2G | 78.9 |
| [mambaout_tiny](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth) | 224 | 26.5M | 4.5G | 82.7 |
| [mambaout_small](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth) | 224 | 48.5M | 9.0G | 84.1 |
| [mambaout_base](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth) | 224 | 84.8M | 15.8G | 84.2 |


#### Usage
We also provide a Colab notebook which runs the steps to perform inference with MambaOut: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing)


## Validation

To evaluate models, run:

```bash
MODEL=mambaout_tiny
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained
```

## Train
We use batch size of 4096 by default and we show how to train models with 8 GPUs. For multi-node training, adjust `--grad-accum-steps` according to your situations.


```bash
DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/MambaOut # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


MODEL=mambaout_tiny 
DROP_PATH=0.2


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH
```
Training scripts of other models are shown in [scripts](/scripts/).


## Bibtex
```
@article{yu2024mambaout,
  title={MambaOut: Do We Really Need Mamba for Vision?},
  author={Yu, Weihao and and Wang, Xinchao},
  journal={arXiv preprint arXiv:2405.07992},
  year={2024}
}
```

## Acknowledgment
Weihao was partly supported by Snap Research Fellowship, Google TPU Research Cloud (TRC), and Google Cloud Research Credits program. We thank Dongze Lian, Qiuhong Shen, Xingyi Yang, and Gongfan Fang for valuable discussions.

Our implementation is based on [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [poolformer](https://github.com/sail-sg/poolformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [metaformer](https://github.com/sail-sg/metaformer) and [inceptionnext](https://github.com/sail-sg/inceptionnext).
