# Introspective Deep Metric Learning

The official implementation of **Introspective Deep Metric Learning** on the image classification task. 

## Datasets 
The datasets should be organized in the data folder.
### ImageNet-1K

Download from [here](https://image-net.org/).

Organize the dataset as follows:

```
- imagenet-1k
    |- train
    |   |- n01440764
    |   |   |- n01440764_10026.JPEG
    |   |   |- ...
    |   |- ...
    |-val
    |   |- n01440764
    |   |   |- ILSVRC2012_val_00000293.JPEG
    |   |   |- ...
    |   |- ...
```

### Cifar-10

Download from [here](http://www.cs.toronto.edu/~kriz/cifar.html).

Organize the dataset as follows:

```
- cifar-10-batches-py
    |- batch.meta
    |- data_batch_1
    |- data_batch_2
    |- data_batch_3
    |- data_batch_4
    |- data_batch_5
    |- test_batch
```

### Cifar-100

Download from [here](http://www.cs.toronto.edu/~kriz/cifar.html)

Organize the dataset as follows:

```
- cifar-100-python
    |- train
    |- test
    |- meta
    |- file.txt~
```

## Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy

## Training
We provide the training settings of our IDML framework with (IDML-CutMix) and without (ISM-Baseline) CutMix on three datasets, which improves the performances compared with original structures.

### ImageNet-1K

To train the proposed IDML framework with CutMix on ImageNet-1K with 4 GPUs, run this command:

```
python train.py \
--net_type resnet \
--dataset imagenet \
--batch_size 256 \
--lr 0.1 \
--depth 50 \
--epochs 300 \
--expname ResNet50 \
-j 40 \
--beta 1.0 \
--cutmix_prob 1.0 \
--no-verbose
```

To train the proposed IDML framework without CutMix on ImageNet-1K with 4 GPUs, run this command:

```
python train.py \
--net_type resnet \
--dataset imagenet \
--batch_size 256 \
--lr 0.1 \
--depth 50 \
--epochs 300 \
--expname ResNet50 \
-j 40 \
--beta 1.0 \
--cutmix_prob 0 \
--no-verbose
```

| Method | Backbone | Top-1 Err | Top-5 Err |
|:-:|:-:|:-:|:-:|
| IDML-CutMix | ResNet-50 | 20.96 | 5.53 |
| ISML-Baseline | ResNet-50 | 23.06 | 6.76 |

### Cifar-10

To train the proposed IDML framework with CutMix on Cifar-10 with 2 GPUs, run this command:

```
python train.py \
--net_type pyramidnet \
--dataset cifar10 \
--depth 200 \
--alpha 240 \
--batch_size 64 \
--lr 0.25 \
--expname PyraNet200 \
--epochs 300 \
--beta 1.0 \
--cutmix_prob 0.5 \
--no-verbose
```

To train the proposed IDML framework without CutMix on Cifar-10 with 2 GPUs, run this command:

```
python train.py \
--net_type pyramidnet \
--dataset cifar10 \
--depth 200 \
--alpha 240 \
--batch_size 64 \
--lr 0.25 \
--expname PyraNet200 \
--epochs 300 \
--beta 1.0 \
--cutmix_prob 0 \
--no-verbose
```

| Method | Backbone | Top-1 Err |
|:-:|:-:|:-:|
| IDML-CutMix | ResNet-50 | 3.57 |
| ISM-Baseline | ResNet-50 | 2.68 |

### Cifar-100

To train the proposed IDML framework with CutMix on Cifar-100 with 2 GPUs, run this command:

```
python train.py \
--net_type pyramidnet \
--dataset cifar100 \
--depth 200 \
--alpha 240 \
--batch_size 64 \
--lr 0.25 \
--expname PyraNet200 \
--epochs 300 \
--beta 1.0 \
--cutmix_prob 0.5 \
--no-verbose
```

To train the proposed IDML framework without CutMix on Cifar-100 with 2 GPUs, run this command:

```
python train.py \
--net_type pyramidnet \
--dataset cifar100 \
--depth 200 \
--alpha 240 \
--batch_size 64 \
--lr 0.25 \
--expname PyraNet200 \
--epochs 300 \
--beta 1.0 \
--cutmix_prob 0 \
--no-verbose
```

| Method | Backbone | Top-1 Err | Top-5 Err |
|:-:|:-:|:-:|:-:|
| IDML-CutMix | ResNet-50 | 14.35 | 2.79 |
| ISM-Baseline | ResNet-50 | 15.92 | 3.54 |

## Device 

We tested our code on a Linux machine with 4 Nvidia RTX 3090 GPU cards. 

## Acknowledgment

Our code is based on [CutMix](https://github.com/clovaai/CutMix-PyTorch).

