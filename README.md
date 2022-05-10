# Introspective Deep Metric Learning

This repository is the official implementation of our paper:

>  Introspective Deep Metric Learning
>
> Chengkun Wang\*, [Wenzhao Zheng\*](https://scholar.google.com/citations?user=LdK9scgAAAAJ&hl=en), [Zheng Zhu](http://www.zhengzhu.net/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), and [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)

## Introduction

This paper proposes an introspective deep metric learning (IDML) framework for uncertainty-aware comparisons of images. Conventional deep metric learning methods produce confident semantic distances between images regardless of the uncertainty level. However, we argue that a good similarity model should consider the semantic discrepancies with caution to better deal with ambiguous images for more robust training. To achieve this, we propose to represent an image using not only a semantic embedding but also an accompanying uncertainty embedding, which describe the semantic characteristics and ambiguity of an image, respectively. We further propose an introspective similarity metric to make similarity judgments between images considering both their semantic differences and ambiguities. Our framework attains state-of-the-art performance on the widely used CUB-200-2011, Cars196, and Stanford Online Products datasets for image retrieval. We further evaluate our framework for image classification on the ImageNet-1K, CIFAR-10, and CIFAR-100 datasets, which shows that equipping existing data mixing methods with the proposed introspective metric consistently achieves better results (e.g., +0.44% for CutMix on ImageNet-1K).

## Motivation

![motivation](./motivation.png)

For a semantically ambiguous image, conventional DML explicitly reduces its distance with other intraclass images unaware of the uncertainty.
Differently, the proposed introspective similarity metric provides an alternative way to enlarge the uncertainty level to allow confusion in the network.  

## Performance

### Image Retrieval

For image retrieval, we followed the setting of [ProxyAnchor](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020).

#### CUB-200-2011
| Methods      | Setting  | R@1      | R@2      | NMI      | RP       | M@R      |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- |
| Triplet-SH   | 512R     | 63.6     | 75.5     | 67.9     | 35.1     | 24.0     |
| **IDML-TSH** | **512R** | **65.3** | **76.5** | **69.5** | **36.2** | **25.0** |
| ProxyNCA     | 512R     | 64.6     | 75.6     | 69.1     | 35.5     | 24.7     |
| **IDML-PN**  | **512R** | **66.0** | **76.4** | **70.1** | **36.5** | **25.4** |
| FastAP       | 512R     | 65.1     | 75.4     | 68.5     | 35.9     | 24.1     |
| **IDML-FAP** | **512R** | **66.4** | **76.4** | **69.7** | **36.7** | **25.5** |
| Contrastive  | 512R     | 65.6     | 76.5     | 68.9     | 36.5     | 24.7     |
| **IDML-Con** | **512R** | **67.2** | **77.6** | **71.3** | **37.5** | **25.7** |
| Margin-DW    | 512R     | 65.9     | 77.0     | 69.5     | 36.0     | 24.9     |
| **IDML-MDW** | **512R** | **67.9** | **78.3** | **72.1** | **37.2** | **26.1** |
| Multi-Sim    | 512R     | 67.3     | 78.2     | 72.7     | 36.6     | 25.5     |
| **IDML-MS**  | **512R** | **69.0** | **79.5** | **73.5** | **38.5** | **27.2** |
| ProxyAnchor  | 512R     | 69.0     | 79.4     | 72.3     | 38.5     | 27.5     |
| **IDML-PA**  | **512R** | **70.7** | **80.2** | **73.5** | **39.3** | **28.4** |

#### Cars196
| Methods      | Setting  | R@1      | R@2      | NMI      | RP       | M@R      |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- |
| Triplet-SH   | 512R     | 70.8     | 81.7     | 64.8     | 31.7     | 21.1     |
| **IDML-TSH** | **512R** | **73.7** | **84.0** | **67.3** | **33.8** | **24.1** |
| ProxyNCA     | 512R     | 82.6     | 89.0     | 66.4     | 33.5     | 23.5     |
| **IDML-PN**  | **512R** | **85.5** | **91.3** | **69.0** | **36.1** | **26.4** |
| FastAP       | 512R     | 81.6     | 88.5     | 68.8     | 35.1     | 25.2     |
| **IDML-FAP** | **512R** | **83.9** | **89.9** | **71.9** | **36.5** | **26.7** |
| Contrastive  | 512R     | 82.7     | 89.6     | 69.5     | 35.8     | 25.7     |
| **IDML-Con** | **512R** | **85.5** | **91.5** | **72.5** | **38.8** | **29.0** |
| Margin-DW    | 512R     | 82.6     | 88.7     | 69.3     | 36.4     | 26.5     |
| **IDML-MDW** | **512R** | **86.1** | **91.7** | **73.0** | **39.2** | **29.7** |
| Multi-Sim    | 512R     | 83.3     | 90.9     | 72.2     | 37.4     | 27.4     |
| **IDML-MS**  | **512R** | **86.3** | **92.2** | **74.1** | **40.0** | **30.8** |
| ProxyAnchor  | 512R     | 87.3     | 92.7     | 75.7     | 40.9     | 31.8     |
| **IDML-PA**  | **512R** | **90.6** | **94.5** | **76.9** | **42.6** | **33.8** |


#### Stanford Online Products
| Methods      | Setting  | R@1      | R@10     | NMI      | RP       | M@R      |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- |
| Triplet-SH   | 512R     | 76.5     | 89.1     | 89.7     | 51.3     | 48.4     |
| **IDML-TSH** | **512R** | **77.4** | **89.4** | **90.1** | **51.9** | **49.0** |
| ProxyNCA     | 512R     | 77.0     | 89.1     | 89.5     | 51.9     | 49.0     |
| **IDML-PN**  | **512R** | **78.3** | **90.1** | **89.9** | **53.0** | **49.9** |
| FastAP       | 512R     | 75.9     | 89.2     | 89.7     | 50.1     | 46.8     |
| **IDML-FAP** | **512R** | **76.8** | **89.7** | **90.9** | **50.9** | **47.9** |
| Contrastive  | 512R     | 76.4     | 88.5     | 88.9     | 50.9     | 47.9     |
| **IDML-Con** | **512R** | **77.3** | **89.7** | **90.0** | **51.7** | **48.5** |
| Margin-DW    | 512R     | 78.5     | 89.9     | 90.1     | 53.4     | 50.2     |
| **IDML-MDW** | **512R** | **79.4** | **90.6** | **91.0** | **53.7** | **50.4** |
| Multi-Sim    | 512R     | 78.1     | 90.0     | 89.9     | 52.9     | 49.9     |
| **IDML-MS**  | **512R** | **79.7** | **91.4** | **91.2** | **53.7** | **50.9** |
| ProxyAnchor  | 512R     | 79.5     | 91.1     | 91.0     | 53.7     | 50.5     |
| **IDML-PA**  | **512R** | **81.5** | **92.7** | **92.3** | **54.8** | **51.3** |




### Image Classification

For image classification, we followed the setting of [CutMix](https://github.com/clovaai/CutMix-PyTorch).

#### ImageNet-1K

| Methods          | Backbone      | Top-1 Acc         | Top-5 Acc         |
| ---------------- | ------------- | ----------------- | ----------------- |
| Baseline         | ResNet-50     | 76.32             | 92.95             |
| **ISM-Baseline** | **ResNet-50** | **76.94 (+0.62)** | **93.24 (+0.29)** |
| Mixup            | ResNet-50     | 77.42             | 93.60             |
| **IDML-Mixup**   | **ResNet-50** | **77.95 (+0.53)** | **93.93 (+0.33)** |
| Cutmix           | ResNet-50     | 78.60             | 94.08             |
| **IDML-Cutmix**  | **ResNet-50** | **79.04 (+0.44)** | **94.47 (+0.39)** |

#### Cifar 100
| Methods          | Backbone      | Top-1 Acc         | Top-5 Acc         |
| ---------------- | ------------- | ----------------- | ----------------- |
| Baseline         | ResNet-50     | 83.55             | 96.31             |
| **ISM-Baseline** | **ResNet-50** | **84.08 (+0.53)** | **96.46 (+0.17)** |
| Mixup            | ResNet-50     | 84.22             | 95.96             |
| **IDML-Mixup**   | **ResNet-50** | **84.59 (+0.37)** | **96.79 (+0.83)** |
| Cutmix           | ResNet-50     | 85.53             | 97.03             |
| **IDML-Cutmix**  | **ResNet-50** | **85.65 (+0.12)** | **97.21 (+0.18)** |

#### Cifar 10
| Methods          | Backbone      | Top-1 Acc         |
| ---------------- | ------------- | ----------------- |
| Baseline         | ResNet-50     | 96.15             |
| **ISM-Baseline** | **ResNet-50** | **96.43 (+0.28)** |
| Mixup            | ResNet-50     | 96.91             |
| **IDML-Mixup**   | **ResNet-50** | **97.13 (+0.22)** |
| Cutmix           | ResNet-50     | 97.12             |
| **IDML-Cutmix**  | **ResNet-50** | **97.32 (+0.20)** |



## Citation

If you find this project useful in your research, please cite:

````
@article{wang2022introspective,
    title={Introspective Deep Metric Learning},
    author={Wang, Chengkun and Zheng, Wenzhao and Zhu, Zheng and Zhou, Jie and Lu, Jiwen},
    journal={arXiv preprint arXiv:2205.04449},
    year={2022}
}
````
