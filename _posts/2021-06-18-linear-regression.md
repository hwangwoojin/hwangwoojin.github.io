---
layout: post
title: 선형회귀
date:   2021-06-18 14:00:00 +0900
description: 파이썬 코드를 통해 선형 회귀 문제를 해결해보자
categories: python
---

이 글에서는 기본적으로 `matplotlib`, `seaborn`, `numpy`를 사옹합니다.

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
```

## 직선함수 모델링

직선함수는 일반적으로 $$y = ax + b$$ 형태를 가지는 함수를 말합니다. 여기서 $$a$$는 기울기(slope)가 되고 $$b$$는 y절편(intercept) 가 됩니다. 먼저 직선 함수 형태의 데이터를 생성하도록 하겠습니다.

```python
rng = np.random.RandomState(1)

# x를 0~10 사이의 값으로 50개 생성
x = 10 * rng.rand(50)

# y = 2x - 5 + <노이즈값> 으로 50개 생성
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y)
```

약간의 노이즈값을 가지는 $$x, y$$ 데이터를 생성하였습니다. `Scikit-Learn`의 `LinearRegression`을 사용하면 이 데이터만을 가지고 이를 가장 잘 표현하는 직선을 찾을 수 있습니다.

```python
from sklearn.linear_model import LinearRegression

# 직선함수 모델
model = LinearRegression(fit_intercept=True)

# 모델을 학습한다.
model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)

# 형식을 맞추기 위해 차원을 하나 증가
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
```

이 코드를 통해 $$x, y$$ 데이터를 통해 모델링한 직선 함수를 그래프로 확인할 수 있습니다. 기울기와 y절편은 아래와 같이 구할 수 있습니다.

```python
print(model.coef_[0]) # 2.027208810360695
print(model.intercept_) # -4.998577085553202
```

## 다차원 함수 모델링

다차원 함수란 $$y = a + bx1 + cx2 + dx3 + ...$$ 형태의 함수를 말합니다. 이것도 직선함수와 유사하게 모델링할 수 있습니다. 아래는 $$y = a + bx1 + cx2 + dx3$$ 함수를 모델링하는 코드입니다.

```python
rng = np.random.RandomState(1)

# (100, 3) 차원으로 데이터 생성
X = 10 * rng.rand(100, 3)

# y = 0.5 + 1.5x + -2x + 1x + <노이즈> 를 행렬 곱으로 생성
y = 0.5 + np.dot(X, [1.5, -2., 1.]) + rng.randn(100)

model.fit(X, y)
```

기울기와 y절편은 다음과 같습니다.

```python
print(model.intercept_) # 0.6562333465768129
print(model.coef_) # [ 1.48159542 -1.97622428  0.97258042]
```

## 다항 기저함수

$$y = a + bx + cx^2 + ...$$와 같은 경우, 기존 선형 모델링으로는 해결할 수 없는것을 확인할 수 있습니다. 이 경우 기저함수(basis function)를 사용할 수 있습니다. 다항 기저함수(Polynomial Basis Function)란 $$y = a + bx1 + cx2 + ...$$에서 기저함수 $$fn(x) = x^n$$을 사용하는 방법입니다. 이 경우 위 함수는 결국 $$y = a + bx + cx^2 + ...$$로 나타나게 되며 이 모델은 여전히 계수에 관해서는 선형함수가 됩니다.

다항 기저함수는 `Scikit-Learn`의 `PolynomialFeatures`를 통해 구현할 수 있습니다.

```python
from sklearn.preprocessing import PolynomialFeatures

x = np.array([2, 3, 4])

# transformer를 통해 3차수까지 확장한다.
poly = PolynomialFeatures(3, include_bias=False)

# x를 [x, x^2, x^3] 형태로 변환한다.
poly.fit_transform(x[:, None])
```

이 경우 $$x$$와 $$poly$$는 각각 다음과 같은 값을 가지게 됩니다.

```python
x
# array([ 2.,  3.,  4.])

poly
# array([[ 2.,  4.,  8.],
#        [ 3.,  9., 27.],
#        [ 4., 16., 64.]])
```

이를 선형모델에 적용할 수 있습니다. 먼저 이를 위해 sine 함수를 사용해서 데이터를 생성했다고 가정하겠습니다.

```python
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)

# y = sinx + 0.1<노이즈>
y = np.sin(x) + 0.1 * rng.randn(50)
plt.scatter(x, y)
```

이를 7차원 변환을 적용한 선형회귀에 적용하여 해결해보도록 하겠습니다. 아래 코드에서는 `make_pipeline`을 통해 변환과 회귀를 한번에 적용하였습니다.

```python
# pipeline은 여러 일을 동시에 처리할 수 있도록 한다.
from sklearn.pipeline import make_pipeline

# 데이터를 7차원 변환한 후 선형회귀를 적용한다.
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
```

## 가우시안 기저함수

기저함수를 위에서 설명한 다항 기저함수 말고도 가우시안 기저함수를 사용할 수도 있습니다. 가우시안 기저함수는 $$exp\{-(x-u_j)^2/2s^2\}$$로 정의되는 함수입니다. 가우시안 기저합수는 주어진 데이터를 여러 개의 가우시안 기저함수들의 합으로 표현합니다. 이때 $$u_j$$는 함수의 위치를 결정하고 $$s$$는 폭을 결정합니다.

가우시안 기저함수는 `Scikit-Learn`에 포함되어 있지 않으므로 직접 구현해야 합니다.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)
    
gauss_model = make_pipeline(GaussianFeatures(20),
                            LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10)
```

## 규제화

너무 많은 기저함수를 사용하면 데이터의 노이즈까지 학습하는 과대적합(over-fitting) 현상이 일어날 수 있습니다.

```python
def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    
    if title:
        ax[0].set_title(title)

    ax[1].plot(model.steps[0][1].centers_,
               model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location',
              ylabel='coefficient',
              xlim=(0, 10))
    
model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)
```

이 코드는 가우시안 기저함수를 30개나 사용하여 과대적합 현상이 일어나는 코드입니다. 과대적합이 발생한 코드의 그래프에서는 인접한 기저함수들의 값이 극단으로 가면서 서로 상쇄하는 현상이 일어납니다.

이 문제를 해결하기 위해서는 큰 계수값에 어느정도 규제를 적용할 필요가 있습니다. 이 방법을 **규제화(Regularization)**라고 한다.

## 리지 회귀

**리지회귀(Lidge Regression)**는 $$P = a(N_1^2 + N_2^2 + ...)$$로 정의됩니다. 이때 $$a$$는 규제의 정도를 결정합니다. $$a$$값이 0에 가까울수록 일반적인 선형회귀모델이 되고 무한대로 증가하면 데이터는 모델에 영향을 주지 않게 됩니다.

`Scikit-Learn`의 `Ridge`를 통해 구현할 수 있습니다.

```python
from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge Regression')
```

## 라쏘 회귀

**라쏘회귀(Lasso Regression)**는 $$P = a(\vert N_1\vert + \vert N_2\vert + ...)$$로 정의됩니다. 계수들의 절대값의 합을 제한하는 방식으로 많은 계수들을 0으로 설정하게 되어 희소한 모델을 생성합니다.

`Scikit-Learn`의 `Lasso`를 통해 구현할 수 있다.

```python
from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso Regression')
```

## SGD

**SGD(Stochastic Gradient Descent)**는 반복을 통해 모델의 그레디언트를 결정하는 방법입니다.

`Scikit-Learn`의 `SGDRegressor`를 통해 구현할 수 있다.

```python
from sklearn.linear_model import SGDRegressor
model = make_pipeline(GaussianFeatures(30),
                      SGDRegressor(max_iter=100, tol=1e-8, alpha=0))
basis_plot(model)
```