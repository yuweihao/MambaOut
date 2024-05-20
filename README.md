# [MambaOut: Do We Really Need Mamba for Vision?](https://arxiv.org/abs/2405.07992)

<p align="center">
<a href="https://arxiv.org/abs/2405.07992" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2405.07992-b31b1b.svg?style=flat" /></a>
<a href="https://huggingface.co/spaces/whyu/MambaOut" alt="Hugging Face Spaces">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" /></a>
<a href="https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

<p align="center"><em>In memory of Kobe Bryant</em></p>

> "What can I say, Mamba out." — *Kobe Bryant, NBA farewell speech, 2016*

<p align="center">
<img src="https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mamba_out.png" width="400"> <br>
<small>Image credit: https://www.ebay.ca/itm/264973452480</small>
</p>


This is a PyTorch implementation of MambaOut proposed by our paper "[MambaOut: Do We Really Need Mamba for Vision?](https://arxiv.org/abs/2405.07992)". 


![MambaOut first figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_first_figure.png)
Figure 1: (a) Architecture of Gated CNN and Mamba blocks (omitting Normalization and shortcut). The Mamba block extends the Gated CNN with an additional state space model (SSM). As will be conceptually discussed in Section 3, SSM is not necessary for image classification on ImageNet. To empirically verify this claim, we stack Gated CNN blocks to build a series of models named MambaOut.(b) MambaOut outperforms visual Mamba models, e.g., Vision Mamhba, VMamba and PlainMamba, on ImageNet image classification. 

<br>

![MambaOut second figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_second_figure.png)
Figure 2: The mechanism illustration of causal attention and RNN-like models from memory perspective, where $x_i$ denotes the input token of $i$-th step. (a) Causal attention stores all previous tokens' keys $k$ and values $v$ as memory. The memory is updated by continuously adding the current token's key and value, so the memory is lossless, but the downside is that the computational complexity of integrating old memory and current tokens increases as the sequence lengthens. Therefore attention can effectively manage short sequences but may encounter difficulties with longer ones. (b) In contrast, RNN-like models compress previous tokens into fixed-size hidden state $h$, which serves as the memory. This fixed size means that RNN memory is inherently lossy, which cannot directly compete with the lossless memory capacity of attention models. Nonetheless, **RNN-like models can demonstrate distinct advantages in processing long sequences,  as the complexity of merging old memory with current input remains constant, regardless of sequence length.**

<br>

![MambaOut third figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_third_figure.png)
Figure 3: (a) Two modes of token mixing. For a total of $T$ tokens, the fully-visible mode allows token $t$ to aggregate inputs from all tokens, i.e., $ \{x_i\}_{i=1}^{T} $, to compute its output $y_t$. In contrast, the causal mode restricts token $t$ to only aggregate inputs from preceding and current tokens $ \{x_i\}_{i=1}^{t} $. By default, attention operates in fully-visible mode but can be adjusted to causal mode with causal attention masks. RNN-like models, such as Mamba's SSM, inherently operate in causal mode due to their recurrent nature. (b) **We modify the ViT's attention from fully-visible to causal mode and observe performance drop on ImageNet, which indicates causal mixing is unnecessary for understanding tasks.**



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
| [mambaout_kobe](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth)\* | 224 | 9.1M | 1.5G | 80.0 |
| [mambaout_tiny](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth) | 224 | 26.5M | 4.5G | 82.7 |
| [mambaout_small](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth) | 224 | 48.5M | 9.0G | 84.1 |
| [mambaout_base](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth) | 224 | 84.8M | 15.8G | 84.2 |

\* [Kobe Memorial Vision](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019) with 24 Gated CNN blocks. MambaOut-Kobe achieves really competitive performance, surpassing ResNet-50 and ViT-S with much fewer parameters and FLOPs. For example, MambaOut-Kobe outperforms ResNet-50 (ResNet strikes back) by 0.2% accuracy with only 36% parameters and MACs.

#### Usage
We also provide a Colab notebook which runs the steps to perform inference with MambaOut: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing).

## Gradio demo
A web demo is shown at [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/whyu/MambaOut). You can also easily run gradio demo locally. Besides PyTorch and timm==0.6.11, please install gradio by `pip install gradio`, then run
```bash
python gradio_demo/app.py
```

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


## Tutorial to compute Transformer FLOPs
This [tutorial](https://github.com/yuweihao/MambaOut/issues/210) shows how to compute FLOPs of a Transformer (Equation 6 in the paper). Welcome feedback, and I will continually improve it.

## Bibtex
```
@article{yu2024mambaout,
  title={MambaOut: Do We Really Need Mamba for Vision?},
  author={Yu, Weihao and Wang, Xinchao},
  journal={arXiv preprint arXiv:2405.07992},
  year={2024}
}
```

## Acknowledgment
Weihao was partly supported by Snap Research Fellowship, Google TPU Research Cloud (TRC), and Google Cloud Research Credits program. We thank Dongze Lian, Qiuhong Shen, Xingyi Yang, and Gongfan Fang for valuable discussions.

Our implementation is based on [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [poolformer](https://github.com/sail-sg/poolformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [metaformer](https://github.com/sail-sg/metaformer) and [inceptionnext](https://github.com/sail-sg/inceptionnext).
