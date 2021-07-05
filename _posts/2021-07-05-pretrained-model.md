---
layout: post
title: torchvision 모델
date:   2021-07-05 07:00:00 +0900
description: torchvision 에서 사용할 수 있는 유명한 신경망들에 대해 알아보자
categories: 프로그래머스-인공지능-데브코스
---

> [pytorch 공식문서](https://pytorch.org/vision/stable/models.html)를 참고하여 작성하였습니다.

학습을 진행할 때 직접 간단한 모델을 정의해서 사용해도 되지만, 이미 높은 성능을 가진 모델을 가져와서 사용하는 것도 좋은 방법입니다.

이 글에서는 `torchvision` 라이브러리에서 여러 모델들을 가져와서 성능을 측정해보도록 하겠습니다. 다음 라이브러리를 사용하였습니다.

```python
import torchvision.models as models
```

## AlexNet

**AlexNet**은 2012년에 영상 인식 대회 ILSVRC에서 우승한 CNN 모델입니다. 5개의 컨볼루션층과 3개의 fc층으로 구성하였으며, max pooling을 할 때 stride를 적게 설정해서 필터를 겹치도록 하여 성능을 향상시켰습니다. 이 외에도 ReLU, Dropout, Data augmentation 등 지금은 보편적으로 사용하지만 당시에는 생소한 방법들을 사용하였습니다.

```python
alexnet = models.alexnet()
```

만약 가중치가 이미 학습된 모델을 가져오고 싶다면 `pretrained=True` 를 사용하면 됩니다.

```python
alexnet = models.alexnet(pretrained=True)
```

ImageNet 데이터셋에서 AlexNet의 정확도는 다음과 같이 알려져 있습니다.

```
ACC@1: 56.522
ACC@5: 79.066
```

## VGG

**VGG**는 옥스포드 VGG 연구팀에서 개발한 모델로, 2014년에 ILSVRC에서 준우승한 모델입니다. 컨볼루션 필터를 제일 작은 3x3 크기로 설정하고, 층의 깊이만 깊게 설정한 모델입니다. VGG16, VGG19 등은 각각 16개, 19개의 층을 깊게 쌓아 만들었다는 것을 의미합니다.

3x3 필터를 2번 사용하는 것은 5x5 필터를 한번 사용하는 것과 같고, 3x3 필터를 3번 사용하는 것은 7x7 필터를 한번 사용하는 것과 같은 효과를 가집니다. 그러나 3x3 필터를 여러번 사용하는 것은 파라미터를 크게 줄일 수 있는 장점이 있고 VGG는 이를 활용하였다는 특징이 있습니다.

```python
vgg16 = models.vgg16()
```

ImageNet에서 VGG의 정확도는 다음과 같이 알려져 있습니다.

```
VGG-11
ACC@1: 69.020
ACC@5: 88.628

VGG-13
ACC@1: 69.928
ACC@5: 89.246

VGG-16
ACC@1: 71.592
ACC@5: 90.382

VGG-19
ACC@1: 72.376
ACC@5: 90.876
```

## ResNet

**ResNet**은 마이크로소프트에서 개발한 2015년에 ILSVRC에서 우승을 차지한 모델입니다. VGG를 기반으로 하였고, 152개의 아주 깊은 층을 사용하였다는 특징이 있습니다.

일반적으로 층을 아주 깊게 쌓으면 제대로 학습이 되지 않는 문제가 존재합니다. ResNet은 이를 해결하기 위해 지름길을 사용해서 입력값을 전달합니다. 그리고 residual block을 사용하여 잔차를 최소화하는 방향으로 학습을 수행합니다.

```python
resnet18 = models.resnet18()
```

ImageNet 데이터셋에서 ResNet의 정확도는 다음과 같습니다.

```
ResNet-18
ACC@1: 69.758
ACC@5: 89.078

ResNet-34
ACC@1: 73.314
ACC@5: 91.420

ResNet-50
ACC@1: 76.130
ACC@5: 92.862

ResNet-101
ACC@1: 77.374
ACC@5: 93.546

ResNet-152
ACC@1: 78.312
ACC@5: 94.046
```

## DenseNet

**DenseNet**은 ResNet을 기반으로 해서 만든 모델입니다. 컨볼루션 층이 입력층이나 출력층에 가까울수록 더 정확하다는 점에 착안해서 모든 층을 서로 연결한 것이 특징입니다.

```python
densenet = models.densenet161()
```

ImageNet 데이터셋에서 DenseNet의 정확도는 다음과 같습니다.

```
Densenet-121
ACC@1: 74.434
ACC@5: 91.972

Densenet-169
ACC@1: 75.600
ACC@5: 92.806

Densenet-201
ACC@1: 76.896
ACC@5: 93.370

Densenet-161
ACC@1: 77.138
ACC@5: 93.560
```
