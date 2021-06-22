---
layout: post
title: 확률분포
date:   2021-06-18 14:00:00 +0900
description: 빈도주의 관점과 베이지안 관점을 통해 확률분포를 구하자
categories: math
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
