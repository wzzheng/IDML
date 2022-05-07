# Introspective Deep Metric Learning

This repository is the official implementation of **Introspective Deep Metric Learning** on the image retrieval task. 

## Datasets 
The datasets should be organized in the data folder.
### CUB-200-2011

Download from [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

Organize the dataset as follows:

```
- CUB_200_2011
    |- images
    |   |- 001.Black_footed_Albatross
    |   |   |- Black_Footed_Albatross_0001_796111
    |   |   |- ...
    |   |- ...
```

### Cars196

Download from [here](http://ai.stanford.edu/~jkrause/cars/car_dataset.html).

Organize the dataset as follows:

```
- cars196
    |- car_ims
    |   |- image000001
    |   |- ...
    |- cars_annos.mat
```

### Stanford Online Products

Download from [here](http://cvgl.stanford.edu/projects/lifted_struct/)

Organize the dataset as follows:

```
- Standford_Online_Products
    |- bicycle_final
    |   |- image0
    |   |- ...
    |- ...
    |- Ebay_train.txt
    |- Ebay_test.txt
```

## Requirements
- Python3
- PyTorch (>1.0)
- NumPy
- wandb

## Training
We provide the training settings of our IDML framework with the ProxyAnchor loss on three datasets, which achieves state-of-the-art performances compared with previous methods.

### CUB-200-2011

To train the proposed IDML framework using the ProxyAnchor loss on CUB200 in the paper, run this command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--gpu -1 \
--loss Proxy_Anchor \
--model resnet50 \
--embedding-size 512 \
--batch-size 120 \
--lr 6e-4 \
--dataset cub \
--warm 5 \
--bn-freeze 1 \
--lr-decay-step 5
```

| Method | Backbone | R@1 | R@2 | R@4 | NMI | RP | MAP@R |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| IDML-PA | ResNet-50 | 70.7 | 80.2 | 87.9 | 73.5 | 39.3 | 28.4 |

### Cars196

To train the proposed IDML framework using the ProxyAnchor loss on CUB200 in the paper, run this command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--gpu -1 \
--loss Proxy_Anchor \
--model resnet50 \
--embedding-size 512 \
--batch-size 120 \
--lr 2.5e-4 \
--dataset cars \
--warm 5 \
--bn-freeze 1 \
--lr-decay-step 10
```

| Method | Backbone | R@1 | R@2 | R@4 | NMI | RP | MAP@R |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| IDML-PA | ResNet-50 | 90.6 | 94.5 | 97.1 | 76.9 | 42.6 | 33.8 |

### Stanford_Online_Products

To train the proposed IDML framework using the ProxyAnchor loss on SOP in the paper, run this command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--gpu -1 \
--loss Proxy_Anchor \
--model resnet50 \
--embedding-size 512 \
--batch-size 120 \
--lr 6e-4 \
--dataset SOP \
--warm 5 \
--bn-freeze 1 \
--lr-decay-step 10 \
--lr-decay-gamma 0.25
```

| Method | Backbone | R@1 | R@10 | NMI | RP | MAP@R |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| IDML-PA | ResNet-50 | 81.5 | 92.7 | 92.3 | 54.8 | 51.3 |

## Device 

We tested our code on a linux machine with 8 Nvidia RTX 2080ti GPU cards. 
