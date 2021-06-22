---
layout: post
title: 확률분포
date:   2021-06-22 17:00:00 +0900
description: 빈도주의 관점과 베이지안 관점을 통해 확률분포를 구하자
categories: 프로그래머스-인공지능-데브코스
---

## 밀도 추정

**밀도 추정(Density estimation)**이란 $$N$$개의 관찰데이터 $$x_1,x_2,...x_N$$가 주어졌을 때 분포함수 $$p(x)$$를 찾는 것을 말합니다. 정확한 분포함수를 알고있다면 이로부터 모든 것을 할 수 있지만, 실제로 정확한 분포함수를 알아내기는 어렵습니다. 따라서 $$p(x)$$를 추정해야 합니다.

분포를 추정하기 위해 파라미터 값을 직접 구하는 **빈도주의** 방법을 사용하거나, 파라미터의 사전확률을 가정하여 사후확률을 구하는 **베이지안** 방법을 사용합니다. 분류문제에서는 주로 $$p(t\vert x)$$ 또는 $$p(C\vert x)$$를 추정합니다. 이 방법을 통해 분포의 파라미터를 찾았다면 이를 통해 $$t, C$$를 예측할 수 있게 됩니다.

### 이항변수

**이항변수(Binary Variable)**는 이항변수는 동전던지기와 같이 단 두가지의 사건만 존재하는 확률변수입니다.

빈도주의 방법을 사용하여 이항변수의 분포인 **이항분포(Binary distribution)**를 추정하는 방법은 다음과 같습니다. 다음은 $$x\in\{0,1\}$$일 때의 확률과 분포 예시입니다.

$$p(x = 1\vert \mu) = \mu, p(x=0\vert\mu)=1-\mu$$

$$Bern(x\vert\mu)=\mu^x(1-\mu)^{1-x}$$

위에서 확률을 **베르누이 분포(Bernoulli distribution)**로 표현하였습니다. 베르누이 분포의 기댓값과 분산은 다음과 같은 값을 가지게 됩니다.

$$E[x]=\mu, var[x]=\mu(1-\mu)$$

이항변수에서 다음과 같은 우도함수(likelihood function)를 만들 수 있습니다.

$$p(D\vert\mu)=\Pi_{n=1}^Np(x_n\vert\mu)=\Pi_{n=1}^N\mu^2(1-\mu)^{1-x_n}$$

로그우도함수를 구하면 다음과 같습니다.

$$\ln p(D\vert\mu)=\sum_{n=1}^N\ln p(x_n\vert\mu)=\sum_{n=1}^N\{x_n\ln\mu + (1-x_n)\ln(1-\mu)\}$$

따라서 우도함수 또는 로그우도함수를 최대화시키는 값으로 $$\mu$$값을 최적화할 수 있게 됩니다. 즉 위 값을 $$\mu$$에 대해서 미분한 후 0으로 놓고 풀면 됩니다. 이 때 구해지는 $$\mu$$의 **최대우도 추정치(maximum likelihood estimate)**는 다음과 같습니다.

$$\mu^ML = \frac{m}{N}$$

이 때 $$m$$은 $$x=1$$일 때의 관찰된 값이 됩니다. 일반적으로 $$N$$이 작은 경우, 최대우도 추정치 값은 과적합된 결과를 얻을 수 있다고 알려져 있습니다.

베이지안 관점에서 베르누이 시행의 반복은 $$x_1,x_2,...x_N$$을 모두 각각의 확률변수로 보게 됩니다. 이항변수의 경우, $$x$$가 1인 경우를 몇번 관찰했는지가 하나의 확률변수 $$m$$이 되며, 0인 경우는 $$N-m$$이 됩니다. 따라서 우도함수 또한 $$x_1,x_2,...,x_N$$이 아니라 변수 $$m$$ 하나로 표현가능하게 됩니다.

이항분포를 $$D=\{x_1,x_2,...,x_N\}$$일 때 이항변수 $$x$$가 1인 경우를 $$m$$번 관찰할 확률로 정의했을 때 분포는 다음과 같습니다.

$$Bin(n\vert N,\mu)=\binom{N}{m}\mu^m(1-\mu)^{N-m}$$

$$\binom{N}{m}=\frac{N!}{(N-m)!m!}$$

$$E[m]=N\mu, var[m]=N\mu(1-\mu)$$

이를 켤레사전분포(conjugate prior)를 사용하여 표현할 수 있습니다. 이때 나타나는 분포를 **베타분포(Beta distribution)**라고 합니다. 베타분포는 $$\mu$$에 대한 분포이자, 정규화된 분포입니다.

$$Beta(\mu\vert a,b)=\frac{\gamma(a+b)}{\gamma(a)\gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$

이 때 감마함수 $$\gamma(x)$$는 다음과 같이 정의됩니다.

$$\gamma(x)=\int_0^\inf \mu^{x-1}e^{-\mu}d\mu$$

여기서 감마함수는 $$\gamma(n)=(n-1)!$$와 같이 정의되어 있습니다. 이는 기존 이항분포에서 사용한 팩토리얼을 실수로 확장시키는 성질을 가지고 있습니다.

베타분포의 기댓값과 분산은 다음과 같이 구할 수 있습니다.

$$E[\mu]=\frac{a}{a+b}, var[\mu]=\frac{ab}{(a+b)^2(a+b+1)}$$

베이즈 확률을 사용하여 $$\mu$$의 사후확률을 구하면 다음과 같습니다.

$$p(\mu\vert m,N-m,a,b) \propto p(m,N-m\vert\mu)p(\mu\vert a,b)$$

이 때 우도 $$p(m,N-m\vert\mu)$$는 이항분포를 통해서, 사전확률 $$p(\mu\vert a,b)$$은 베타분포를 통해서 구할 수 있습니다.

$$p(\mu\vert m,N-m,a,b) = \frac{Bin(m\vert N,\mu)Beta(\mu\vert a,b)}{\int_0^1Bin(m\vert N,\mu)Beta(\mu\vert a,b)d\mu}$$

이를 풀면 다음과 같습니다.

$$\frac{\gamma(m+a+(N-m)+b)}{(\gamma(m+a)\gamma(N-m+b))}\mu^{m+a-1}(1-\mu)^{N-m+b=1}$$

만약 $$x=1$$이 주어졌을 때 예측분포(predictive distribution)는 다음과 같이 구할 수 있습니다.

$$p(x=1\vert D) = \int_0^1p(x=1\vert\mu)p(\mu\vert D)d\mu = \int_0^1\mu p(\mu\vert D)d\mu = E[\mu\vert D]$$

$$p(x=1\vert D) = \frac{m+a}{m+a+(N-m)+b}$$

## 다항변수

**다항변수(Multinomial variable)**는 여러 사건 또는 차원이 존재할 수 있는 확률변수를 말합니다. 이 확률변수는 $$K$$차원의 벡터 $$x$$로 나타낼 수 있으며, 이 때 하나의 원소만 1이고 나머지는 0인 성질을 가집니다.

빈도주의 방법에서 다항변수의 확률은 다음과 같이 나타낼 수 있습니다.

$$p(x\vert \mu)=\Pi_{k=1}^{K}\mu_k^{x_k}, \sum_k\mu_k=1$$

여기서 $$x$$의 기댓값은 다음과 같습니다.

$$E[x\vert\mu] = \sum_xp(x\vert\mu)=\mu_1,\mu_2,...,\mu_M)^T=\mu$$

$$x$$값을 $$N$$번 관찰한 결과 $$D=\{x_1,x_2,...,x_N\}$$가 주어졌을 때 우도함수는 다음과 같습니다.

$$p(D\mu)=\Pi_{n=1}^N\Pi_{k=1}^{K}\mu_k^{x_nk} = \Pi_{k=1}^K\mu_k^{m_k}$$

$$p(D\vert\mu)=\mu^m(1-\mu)^{N-m}$$

최대우도추정치를 구하기 위해서는 $$\mu_k$$의 합이 1이 된다는 조건하에서 로그우도를 최대화시키는 $$\mu_K$$를 구해야 합니다. 이를 구하기 위해서 라그랑즈 승수(Lagrange multiplier) $$\lambda$$를 사용해서 다음을 최대화시키면 됩니다.

$$\sum_{k=1}^Km_k\ln\mu_k + \lambda\big(\sum_{k=1}^K\mu_k-1\big)$$

이를 $$\mu_k$$에 관해 미분하면 다음과 같은 최대우도 추정치를 구할 수 있습니다.

$$\mu_k^{ML}=\frac{m_k}{N}$$

베이지안 관점에서 **다항분포(Multinomial distribution)**는 이항분포를 일반화한 형태가 됩니다.

$$Mult(m_1,m_2,...,m_k\vert\mu,N)=\binom{N}{m_1m_2...m_k}\Pi_{k=1}^{K}\mu_k^{m_k}$$

$$\binom{N}{m_1m_2...m_k}=\frac{N!}{m_1!m_2!...m_k!}$$

$$\sum_{k=1}^Km_k=N$$

켤레사전분포를 위해 **디리큘레 분포(Dirichlet distribution)**를 사용할 수 있습니다. 이 분포는 이항분포를 일반화한 형태이며, 베타분포와 마찬가지로 정규화된 분포입니다.

$$Dir(\mu\vert\alpha)=\frac{\gamma(\alpha_0)}{\gamma(\alpha_1)...\gamma(\alpha_k)}\Pi_{k=1}^K\mu_k^{\alpha_k-1}$$

$$\alpha_0 = \sum_{k=1}^K\alpha_k$$

디리큘레 분포를 통해 $$\mu$$의 사후확률을 다음과 같이 구할 수 있습니다.

$$p(\mu\vert D,\alpha)=Dir(\mu\vert\alpha+m) = \frac{\gamma(a_0+N)}{\gamma(\alpha_1+m_1)...\gamma(\alpha_k+m_k)}\Pi_{k=1}^K\mu_k^{\alpha_k+m_k-1}$$

$$m=(m_1,m_2,...,m_k)^T$$

## 가우시안 분포

가우시안 분포가 무엇인지에 대해서는 이전 글에서 살펴본 적이 있습니다. 이 분포는 단일변수 $$x$$에 대해 다음과 같이 정의됩니다.

$$N(x\vert \mu, \sigma) = \frac{1}{(2\pi \sigma^2)^{1/2}}exp\big\{-\frac{1}{2\sigma^2}(x-\mu)^2\big\}$$

표준편차를 분산에 대해 표현하면 다음과 같습니다.

$$N(x\vert \mu, \Sigma) = \frac{1}{(2\pi \Sigma)^{1/2}}exp\big\{-\frac{1}{2\Sigma}(x-\mu)^2\big\}$$

만약 $$x$$가 D차원의 평균 벡터라면 $$\Sigma$$는 $$D * D$$ 크기를 가지는 공분산 행렬이 되며, 식을 다음과 같이 쓸 수 있습니다.

$$N(x\vert \mu, \Sigma) = \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert\Sigma\vert^{1/2}}exp\big\{-\frac{1}{2}(X-\mu)^T\Sigma^{-1}(x-\mu)\big\}$$

이 때 $$\mu, \Sigma$$는 확률밀도함수의 평균과 공분산을 의미합니다. 이 때 공분산을 아래와 같이 나타낼 수 있습니다.

$$U=\begin{bmatrix}u_1^T \\ u_2^T \\ ... \\ u_D^T\end{bmatrix}, A = diag(\lambda_1, \lambda_2,...\lambda_D)$$

$$\Sigma = U^TAU = \sum_{i=1}^D\lambda_iu_iu_i^T$$

이를 통해 이차형식을 구하면 다음과 같습니다.

$$\Sigma^{-1} = \sum_{i=1}^D\frac{1}{\lambda_i}u_iu_i^T$$

$$\Delta^2 = (x-\mu)^T\Sigma^{-1}(x-\mu) = (x-\mu)^T\big(\sum_{i=1}^D\frac{1}{\lambda_i}u_iu_i^T\big)(x-\mu) = \sum_{i=1}^D\frac{y_i^2}{\lambda_i}$$

$$y_i = u_i^T(x-\mu)$$

벡터식으로는 다음과 같이 표현가능합니다.

$$y = U(x-\mu)$$

이 식을 $$y$$를 벡터들 $$\mu_i$$에 의해 기저변환한 것으로 해석할 수도 있습니다. 고유벡터들의 집합 $$U$$에 대하여 $$x-\mu=U^Ty$$라고 한다면 가우시안 분포를 $$U$$를 기저로 하는 좌표계에서의 점들이 됩니다.

가우시안 분포에 의해 생성된 데이터 $$X=(x_1,x_2,...,x_n)^T$$가 주어졌을 때, 최대로그우도 추정을 사용하여 우도함수를 최대화하는 평균값과 공분산값을 구할 수 있습니다. 우도를 최대화하는 평균벡터는 다음과 같이 구할 수 있습니다.

$$\mu_{ML} = \frac{1}{N}\sum_{i=1}^Nx_i=\hat{x}$$

이 때 평균벡터는 생성된 데이터들의 평균값이 됩니다.

최대로그우도 추정을 사용하여 우도를 최대화하는 공분산행렬을 구하면 다음과 같습니다.

$$\Sigma_{ML} = \frac{1}{N}(x_n-\mu)(x_n-\mu)^T$$

만약에 베이지안 방법을 사용한다면 최대우도추정을 통해 $$\mu, \Sigma$$ 값만을 구하는 것과는 다르게 확률분포 자체를 구할 수 있도록 합니다. 이 경우 우도함수와 사전확률, 사후확률은 다음과 같이 구할 수 있습니다.

$$p(x|\mu) = \Pi_{n=1}^Np(x_n\vert\mu)$$

$$p(\mu)=N(\mu\vert\mu_0,\sigma_0^2)$$

$$p(\mu\vert x) = N(\mu\vert\mu_N,\sigma_N^2)$$

## 파이썬 코드로 확률분포 표현하기

파이썬 라이브러리인 `scipy`를 통해 여러 분포들을 생성하고 해당 그래프를 그려보도록 하겠습니다.

이 글에서는 기본적으로 다음과 같은 라이브러리들을 사용하였습니다. 코드는 구글 코랩 환경 기반으로 실행되었습니다.

```python
# for inline plots in jupyter
%matplotlib inline
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
import numpy as np
```

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
