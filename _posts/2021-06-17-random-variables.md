---
layout: post
title:  확률변수
date:   2021-06-17 18:00:00 +0900
description: 확률변수, 누적분포함수, 확률밀도함수, 연속확률변수, 확률변수의 함수
categories: 프로그래머스-인공지능-데브코스
---

**확률변수(Random Variable)**란 표본의 집합 $$S$$의 원소 $$e$$를 실수값 $$X(e) = x$$에 대응시키는 함수입니다. 보통 $$X, Y, Z$$와 같이 대문자로 표현합니다.

앞면을 내야 1점을 얻는 동전던지기를 예로 들면, 표본의 집합은 $$S=\{H,T\}$$, 확률변수는 $$X(H)=1$$, $$X(T)=0$$으로 표현할 수 있습니다. 이 경우 확률 $$P$$는 $$P[X=0]=\frac{1}{2}, P[X=1]=\frac{1}{2}$$가 됩니다.

**누적분포함수(Continuous Random Variables, CDF)**란 $$F(x) = P[X \in (-\infty,x)]$$를 만족하는 함수를 말합니다.

**확률밀도함수(Probability Density Function, PDF)**란 $$F(x)=\int_{-\infty}^{x}{f(t)dt}$$를 만족하는 함수 $$f(x)$$를 만족하는 $$f(x)$$를 말합니다. 이 때 $$X$$를 **연속확률변수(Continuous Random Variables)**라고 합니다. 연속확률변수는 반드시 다음을 만족해야 합니다.

$$p(x) \geq 0, \int_{-\infty}^{\infty}{p(x)} = 1$$

이 확률변수의 함수 또한 확률변수입니다. 예를 들어 $$X$$가 일주일(week)에 대한 확률변수라면, $$Y=7X$$를 통해 매일(day)에 대한 확률변수를 정의할 수 있습니다.

확률변수 $$X$$의 함수 $$Y=g(X)$$와 역함수 $$w(Y)=X$$가 주어졌을 때 다음이 성립한다고 알려져 있습니다.

$$p_y(y) = p_x(x)\big\vert\frac{dx}{dy}\big\vert$$

만약 확률변수가 여러차원을 가지는 경우, 즉 $$X=(x_1,x_2,...x_k), y=(y_1,y_2,...y_k)$$인 경우 야코비안 행렬식 $$J$$를 사용해서 다음과 같이 구할 수 있습니다.

$$p_Y(y_1,y_2,...y_k)=p_X(x_1,x_2,...x_k)\vert J\vert$$

$$J=\begin{vmatrix}\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} & ... & \frac{\partial x_1}{\partial y_k} \\ \frac{\partial x_2}{\partial y_1} & ... & ... & ... \\ ... & ... & ... & ... \\ \frac{\partial x_k}{\partial y_1} & ... & ... & \frac{\partial x_k}{\partial y_k} \end{vmatrix}$$
