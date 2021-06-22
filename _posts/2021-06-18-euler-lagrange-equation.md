---
layout: post
title: 오일러-라그랑주 방정식
date:   2021-06-18 13:00:00 +0900
description: 오일러-라그랑주 방정식에 대해 알아보자.
categories: 프로그래머스-인공지능-데브코스
---

**오일러-라그랑주 방정식(Euler-Lagrange equation)**은 범함수의 최솟값을 구하는데 종종 사용됩니다. 이 방정식은 $$x, y(x), y`(x)$$로 정의되어있는 함수 $$G$$에 대해서 다음과 같습니다.

$$F[y] = \int_a^bG(x, y(x), y'(x))dx$$

$$\frac{\partial F}{\partial y(x)} = \frac{\partial G}{\partial y} - \frac{d}{dx}\frac{\partial G}{\partial y'}$$

오일러-라그랑주 방정식은 회귀문제에서 손실함수의 최솟값을 구하는데 사용됩니다. 회귀문제에서 손실함수는 $$\int\int\{y(x)-t\}^2p(t\vert x)p(x)dxdt$$로 정의됩니다. 이 때 $$\int\{y(x)-t\}^2p(t\vert x)p(x)dxdt$$를 $$G$$라 한다면 다음과 같은 풀이가 가능합니다.

$$\frac{\partial G}{\partial y} = p(x)\frac{\partial}{\partial y}\big(\int_R\{y(x) - t\}^2p(t\vert x)dt\big)$$

이 때 $$\frac{\partial G}{\partial y} = 0$$이라고 하고 식을 $$y$$에 대해서 풀면 최종적으로 다음과 같은 결과를 얻을 수 있습니다.

$$2y(x)-2E_t[t\vert x] = 0$$

$$y(x) = E_t[t\vert x]$$
