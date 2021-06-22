---
layout: post
title: 확률분포 (파이썬)
date:   2021-06-22 13:00:00 +0900
description: 파이썬 코드를 통해 여러 확률분포를 구해보자
categories: 프로그래머스-인공지능-데브코스
---

파이썬 라이브러리인 `scipy`를 통해 여러 분포들을 생성하고 해당 그래프를 그려보도록 하겠습니다.

이 글에서는 기본적으로 다음과 같은 라이브러리들을 사용하였습니다. 코드는 구글 코랩 환경 기반으로 실행되었습니다.

```python
# for inline plots in jupyter
%matplotlib inline
# import matplotlib
import matplotlib.pyplot as plt
# for latex equations
from IPython.display import Math, Latex
# for displaying images
from IPython.core.display import Image
# import seaborn
import seaborn as sns
import numpy as np
```

## 균일분포

**균일분포(Uniform distribution)**는 `scipy.stats.uniform`을 통해 구현할 수 있습니다.

```python
from scipy.stats import uniform

# random numbers from uniform distribution
n = 10000
data_uniform = uniform.rvs(size=n, loc=0, scale=1)

ax = sns.distplot(data_uniform,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')
```

`loc = 0, scale = 1`은 생성할 균일 분포를 0에서 1의 길이만틈만 사용하겠다는 것을 의미합니다. 따라서 여기서는 0과 1사이의 균일분포가 생성됩니다.

## 베르누이 분포

**베르누이 분포(Bernoulli distribution)**는 `scipy.stats.bernoulli`를 통해 구현합니다.

```python
from scipy.stats import bernoulli
data_bern = bernoulli.rvs(size=10000,p=0.8)

ax= sns.distplot(data_bern,
                 kde=False,
                 color="skyblue",
                 hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Bernoulli Distribution', ylabel='Frequency')
```

여기서 파라미터 `p = 0.8`은 베르누이 분포애서 1에 해당하는 빈도를 0.8, 0에 해당하는 빈도를 0.2로 부여하겠다는 것을 의미합니다.

## 베타분포

**베타분포(Beta distrubtion)**는 `scipy.stats.beta`를 통해 구현할 수 있습니다.

```python
from scipy.stats import beta
a, b = 0.1, 0.1
data_beta = beta.rvs(a, b, size=10000)

ax= sns.distplot(data_beta,
                 kde=False,
                 color="skyblue",
                 hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Beta Distribution', ylabel='Frequency')
```

여기서 설정한 `a, b = 0.1, 0.1`은 베타분포에서 감마함수에서의 $$a, b$$ 값을 말합니다. a와 b를 둘다 0.1로 설정한 경우, 대부분의 데이터가 0과 1 근처에서 생성되는 분포가 나타납니다.

## 다항분포

**다항분포(Multinomial distribution)**는 `scipy.stats.multinomial`을 통해 구현합니다.

```python
from scipy.stats import multinomial
data_multinomial = multinomial.rvs(n=1, p=[0.2, 0.1, 0.3, 0.4], size=10000)

for i in range(4):
  print(np.unique(data_multinomial[:,i], return_counts=True))
# (array([0, 1]), array([7960, 2040]))
# (array([0, 1]), array([8952, 1048]))
# (array([0, 1]), array([7006, 2994]))
# (array([0, 1]), array([6082, 3918]))
```

이 다항분포는 하나의 원소가 1이고 나머지는 0인 값을 가지게 되며, 총 원소의 수는 p에 따라 달라집니다. 위의 파라미터 `p=[0.2, 0.1, 0.3, 0.4]`는 0번, 1번, 2번, 3번 확률변수일 확률이 각각 0.2, 0.1, 0.3, 0.4만큼 된다는 것을 말합니다. 이 때 총합은 반드시 1이 되어야 합니다.

쉽게 말해서 값을 생성했을 때 `[1,0,0,0]`,`[0,1,0,0]`,`[0,0,1,0]`,`[0,0,0,1]`일 확률이 각각 0.2, 0.1, 0.3, 0.4가 된다는 의미입니다.

코드에서는 샘플이 2040, 1048, 2994, 3918개가 생성되어 이 분포를 따르는 것을 볼 수 있습니다.
