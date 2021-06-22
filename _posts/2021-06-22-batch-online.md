---
layout: post
title: 배치 학습과 온라인 학습
date:   2021-06-22 21:00:00 +0900
description: 배치학습을 할 때 행렬곱을 위한 메모리가 부족하다면
categories: 프로그래머스-인공지능-데브코스
---

이 글은 다음 파이썬 라이브러리들을 사용하였습니다.

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import numpy.linalg as LA

from sklearn.linear_model import LinearRegression
```

파이썬을 사용하여 $$y = 1.5x_1 + -2x_2 + x_3 + 0.5$$ 형태의 선형회귀 문제를 만들어보도록 하겠습니다.

```python
model = LinearRegression(fit_intercept=True)

N = 1000
M = 3
rng = np.random.RandomState(1)
rng.rand(N, M).shape
X = 10 * rng.rand(N, M)
np.dot(X, [1.5, -2., 1.]).shape

rng = np.random.RandomState(1)
X = 10 * rng.rand(N, M)
y = 0.5 + np.dot(X, [1.5, -2., 1.])

model.fit(X, y)
print(model.intercept_)
# 0.499999999999968

print(model.coef_)
# [ 1.5 -2.   1. ]
```

선형회귀 문제에서는 제곱합 에러함수를 최소화시키기 위해 정규방정식을 구해서 해결했습니다. 이 경우 정규방정식은 $$W_{ML} = (\phi^T\phi)^{-1}\phi^Tt$$로 구할 수 있었습니다. 만약 $$\phi(x)$$가 행렬 $$X$$로 주어졌을 때, 정규방정식을 행렬로 표현하면 다음과 같습니다.

$$(X^TX)^{-1}X^Ty$$

이렇게 정규방정식을 한번에 해결하는 방법을 **배치(batch) 학습**이라고 합니다. 선형회귀 문제에서 다음과 같은 코드로 배치 학습을 수행할 수 있습니다.

```python
LA.inv(X.T@X)@X.T@y
# array([ 1.52825484, -1.96886193,  1.03058857])
```

이 방식의 문제는 행렬 $$X^T, X, X^TX, (X^TX)^{-1}$$ 등을 한번에 다 구해야하기 때문에 행렬의 크기가 커질 경우 사용하는 메모리가 매우 큰 폭으로 커질 수 있다는 것입니다. 현실에 존재하는 많은 상황들의 경우 데이터의 크기가 매우 큰 경우가 많아 보통 이렇게 한번의 계산으로 해결할 수 없습니다.

**온라인(online) 학습**은 행렬을 한번에 구하는 것이 아니라 나누어 구하도록 하여 메모리를 효율적으로 사용할 수 있도록 하는 방법입니다.

```python
A = np.zeros([M, M])
b = np.zeros([M, 1])
for i in range(N):
    A = A + X[i, np.newaxis].T@X[i, np.newaxis]
    b = b + X[i, np.newaxis].T*y[i]
    
LA.inv(A)@b
# array([ 1.52825484, -1.96886193,  1.03058857])
```

**스토캐스틱 경사 하강법(SGD, Stochastic gradient descent)**은 대표적인 온라인 학습 방법입니다. SGD는 기울기(gradient)를 한번에 구하는 것이 아니라 `learning_rate`를 통해 서서히 학습을 할 수 있도록 합니다.

```python
w = np.zeros([M, 1])
learning_rate = 0.001 # learning_rate
epochs = 1000

for i in range(epochs):
    i = i % N
    neg_gradient = (y[i] - w.T@X[i, np.newaxis].T) * X[i, np.newaxis].T
    w += learning_rate * neg_gradient

w
# [[ 1.52825484, -1.96886193,  1.03058857]]
```
