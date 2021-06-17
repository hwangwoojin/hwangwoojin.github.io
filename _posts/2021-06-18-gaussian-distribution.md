---
layout: post
title:  가우시안 분포
date:   2021-06-18 00:00:00 +0900
description: 가우시안 분포에 대해서 알아보자
categories: math
---

**가우시안 분포(Gaussian distribution)**란 다음을 만족하는 분포를 말합니다.

$$N(x\vert \mu, \sigma) = \frac{1}{(2\pi \sigma^2)^{1/2}}exp\big\{-\frac{1}{2\sigma^2}(x-\mu)^2\big\}$$

$$\int_{-\infty}^{\infty}N(x\vert \mu, \sigma^2)dx = 1$$

이 때 아래 식과 같이 모든 구간에서 분포의 합이 1인 경우 이 분포가 **정규화(normalized)**되었다고 합니다. 가우시안 분포는 정규화된 분포입니다. 가우시안 분포의 기댓값은 $$\mu$$이고, 분산은 $$\sigma^2$$임이 알려져 있습니다.

독립적으로 같은 가우시안 분포로부터 $$N$$개의 샘플 $$X = (x_1,x_2,...,x_N)^T$$을 추출했을 때, 모든 가우시안 분포가 독립적이므로 확률은 모든 가우시안을 곱한 값이 됩니다.

$$p(X\vert \mu, \sigma^2) = p(x_1,x_2,...,x_N\vert \mu, \sigma^2)=\prod_{n=1}^{N}N(x_n\vert \mu, \sigma^2)$$

위 식에 로그를 씌우게 되면 곱셈을 간단한 덧셈으로 바꿀 수 있습니다.

$$\ln p(X\vert \mu, \sigma^2) = -\frac{1}{2\sigma^2}\sum_{n=1}^N{(x_n-\mu)}^2 - \frac{N}{2}\ln \sigma^2 - \frac{N}{2}\ln 2\pi$$

이 우도함수 식을 $$\mu$$에 대해 미분하고 0으로 놓고 풀면 최대우도해(Maximum likelihood solution)를 구할 수 있습니다.

$$\frac{\partial}{\partial \mu}\ln p(X\vert \mu, \sigma^2) = \frac{\partial}{\partial \mu}\big\{-\frac{1}{2\sigma^2}\sum_{n=1}^N{(x_n-\mu)}^2 - \frac{N}{2}\ln \sigma^2 - \frac{N}{2}\ln 2\pi\big\}$$

$$ \mu_{ML} = \frac{1}{N}\sum_{n=1}^Nx_n $$

즉 최대우도해를 만들어주는 기댓값은 $$N$$개의 샘플 기댓값들의 평균이 되는 것을 확인할 수 있습니다.

마찬가지로 우도함수 식을 $$\sigma^2$$에 대해 미분하게 되면 다음과 같은 값을 구할 수 있습니다.

$$ \sigma^2_{ML} = \frac{1}{N}\sum_{n=1}^N(x_n-\mu_{ML})^2 $$

이로부터 위에서 구한 기댓값을 통해 분산을 구했을 때 최대우도해를 만들 수 있다는 것을 알 수 있습니다.
