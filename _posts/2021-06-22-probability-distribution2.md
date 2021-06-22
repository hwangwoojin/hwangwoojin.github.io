---
layout: post
title: 확률분포 2
date:   2021-06-22 17:00:00 +0900
description: 빈도주의 관점과 베이지안 관점을 통해 확률분포를 구하자 2
categories: math
---

지난 확률분포 글에서는 이항변수와 다항변수의 확률분포를 구하는 방법에 대해 다루어 보았습니다.

이 글에서는 가우시안 분포에 대해서 다루어 보겠습니다.

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
