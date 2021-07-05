---
layout: post
title: torchvision 전이 학습
date:   2021-07-05 09:00:00 +0900
description: torchvision의 모델을 가져와서 전이 학습을 해보자
categories: 프로그래머스-인공지능-데브코스
---

> [pytorch 공식문서](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)를 참고하여 작성하였습니다.

**전이 학습(Transfer learning)**이란 학습을 할 때 데이터가 부족한 경우, 이미 학습이 완료된 모델을 가져와서 사용하는 것을 말합니다.

전이학습에는 크게 두가지가 존재합니다. 하나는 모델을 가져와서 컨볼루션 층과 fc층 모두를 새로 학습해서 사용하는 방법이고, 두번째는 fc층만 학습해서 사용하는 방법입니다.

두가지 클래스가 있는 데이터셋 분류를 resnet18 모델을 사용해서 전이 학습을 수행하겠습니다.

## 컨볼루션 층과 fc층 둘 다 학습

```python
# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)

model = model.cuda()
...
```

## fc층만 학습

fc층만 학습하기 위해서는 컨볼루션 층의 `backward` 계산을 차단해야 합니다. 이를 위해서는 `requires_grad=False` 파라미터를 사용하면 됩니다.

```python
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)

model = model.cuda()
```
