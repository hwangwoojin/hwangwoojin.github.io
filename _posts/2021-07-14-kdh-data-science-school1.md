---
layout: post
title: 기본 환경 구축
date:   2021-07-14 13:00:00 +0900
description: 데이터 사이언스 스쿨을 위한 기본적인 환경을 구축해보자.
categories: 김도형-데이터-사이언스-스쿨
---

> 책 [김도형의 데이터 사이언스 스쿨 (수학편)](http://www.yes24.com/Product/Goods/82444473)을 참고하였습니다.

> 웹사이트 [데이터 사이언스 스쿨](https://datascienceschool.net)을 참고하였습니다.

여기서는 **김도형의 데이터 사이언스 스쿨**을 위한 기본적인 환경 설정에 대해서 다룹니다.

## 환경

python3

colab notebook

## 파이썬 패키지

```python
# 경고를 무시
import warnings
warnings.simplefilter('ionore')

import matplotlib as mpl
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import sklearn as sk

# matplotlib 설정
mpl.use('Agg')

# seaborn 설정
sns.set()
sns.set_style('whitegrid')
sns.set_color_codes()
```
