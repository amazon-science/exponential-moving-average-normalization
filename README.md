## EMAN: Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning

This is a PyTorch implementation of the [EMAN paper](https://arxiv.org/abs/2101.08482). It supports three popular self-supervised and semi-supervised learning techniques, i.e., [MoCo](https://arxiv.org/abs/1911.05722), [BYOL](https://arxiv.org/abs/2006.07733) and [FixMatch](https://arxiv.org/abs/2001.07685).

If you use the code/model/results of this repository please cite:
```
@inproceedings{cai21eman,
  author  = {Zhaowei Cai and Avinash Ravichandran and Subhransu Maji and Charless Fowlkes and Zhuowen Tu and Stefano Soatto},
  title   = {Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning},
  booktitle = {CVPR},
  Year  = {2021}
}
```


### Install

First, [install PyTorch](https://pytorch.org/get-started/locally/) and torchvision. We have tested on version of 1.7.1, but the other versions should also be working, e.g. 1.5.1.

```bash
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
``` 

Also install other dependencies. 

```bash
$ pip install pandas faiss-gpu
``` 


### Data Preparation

Install ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet), with the standard data folder structure for the torchvision ``datasets.ImageFolder``. Please download the ImageNet [index files](https://eman-cvpr.s3.amazonaws.com/imagenet_indexes.zip) for semi-supervised learning experiments. The file structure should look like:

  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   └── *.jpeg
  │   ├── class2
  │   │   └── *.jpeg
  │   └── ...
  ├── val
  │   ├── class1
  │   │   └── *.jpeg
  │   ├── class2
  │   │   └── *.jpeg
  │   └── ...
  └── indexes
      └── *_index.csv
  ```

### Training

To do self-supervised pre-training of MoCo-v2 with EMAN for 200 epochs, run:
```
python main_moco.py \
  --arch MoCoEMAN --backbone resnet50_encoder \
  --epochs 200 --warmup-epoch 10 \
  --moco-t 0.2 --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /path/to/imagenet
```

To do self-supervised pre-training of BYOL with EMAN for 200 epochs, run:
```
python main_byol.py \
  --arch BYOLEMAN --backbone resnet50_encoder \
  --lr 1.8 -b 512 --wd 0.000001 \
  --byol-m 0.98 \
  --epochs 200 --cos --warmup-epoch 10 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /path/to/imagenet
```

To do semi-supervised training of FixMatch with EMAN for 100 epochs, run:
```
python main_fixmatch.py \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.03 \
  --epochs 100 --schedule 60 80 \
  --warmup-epoch 5 \
  --trainindex_x train_10p_index.csv --trainindex_u train_90p_index.csv \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /path/to/imagenet
```

### Linear Classification and Finetuning

With a pre-trained model, to train a supervised linear classifier on frozen features/weights (e.g. MoCo) on 10% imagenet, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --epochs 50 --schedule 30 40 \
  --eval-freq 5 \
  --trainindex train_10p_index.csv \
  --model-prefix encoder_q \
  --pretrained /path/to/model_best.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /path/to/imagenet
```

To finetune the self-supervised pretrained model on 10% imagenet, with different learning rates for pretrained backbone and last classification layer, run:
```
python main_cls.py \
  -a resnet50 \
  --lr 0.001 --lr-classifier 0.1 \
  --epochs 50 --schedule 30 40 \
  --eval-freq 5 \
  --trainindex train_10p_index.csv \
  --model-prefix encoder_q \
  --self-pretrained /path/to/model_best.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /path/to/imagenet
```

For BYOL, change to ``--model-prefix online_net.backbone``. For the best performance, follow the learning rate setting in Section 5.2 in the paper.


### Models

Our pre-trained ResNet-50 models can be downloaded as following:

| name | epoch | acc@1% IN | acc@10% IN | acc@100% IN | model |
| :---: | :---: | :---: | :---: | :---: | :---: |
| MoCo-EMAN | 200 | 48.9 | 60.5 | 67.7 | [download](https://eman-cvpr.s3.amazonaws.com/models/res50_moco_eman_200ep.pth.tar) |
| MoCo-EMAN | 800 | 55.4 | 64.0 | 70.1 | [download](https://eman-cvpr.s3.amazonaws.com/models/res50_moco_eman_800ep.pth.tar) |
| MoCo-2X-EMAN | 200 | 56.8 | 65.7 | 72.3 | [download](https://eman-cvpr.s3.amazonaws.com/models/res50_2x_moco_eman_200ep.pth.tar) |
| BYOL-EMAN | 200 | 55.1 | 66.7 | 72.2 | [download](https://eman-cvpr.s3.amazonaws.com/models/res50_byol_eman_200ep.pth.tar) |

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
