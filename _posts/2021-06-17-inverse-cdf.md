---
layout: post
title: Inverse CDF
date:   2021-06-17 20:00:00 +0900
description: 밀도함수로부터 실제 샘플을 만들어내고 싶을 때 사용하는 방법
categories: math
---

만약 주어진 밀도함수로부터 몇몇 값을 샘플링하고 싶다면 어떻게 해야할까요?

확률변수 $$X$$가 누적분포함수 $$F_x(x)$$를 가지고, 연속확률분포함수 $$U$$가 $$uniform(0, 1)$$로 정의된다고 할 때, 다음과 같은 확률변수 $$Y$$를 생각할 수 있습니다.

$$Y=F_x^{-1}(U)$$

이때 확률변수 $$Y$$를 정리하면 다음과 같습니다.

$$F_Y(y) = P[Y \leq y] = P[F_x{-1}(U) \leq y] = P[U \leq F_x(y) ] = F_x(y)$$

즉 확률변수 $$Y$$는 확률변수 $$X$$와 동일한 분포를 따르게 됩니다. 이를 **Inverse CDF Method**라고도 합니다. 이 방법은 모든 확률분포의 누적분포함수가 균등분포를 따른다는 성질을 활용하여 밀도함수로부터 실제 샘플을 만들어내고 싶을 때 사용합니다.

예를들어, 반지름이 $$r$$인 원으로부터 랜덤한 샘플링을 하기 위해서는 원점으로부터 거리가 $$d$$보다 작을 확률에 대한 누적분포함수 $$F(d)$$의 역함수를 구하면 됩니다.

$$F(d)=\frac{\pi d^2}{\pi r^2}=\frac{d^2}{r^2}$$

$$F^{-1}(d)=r\sqrt{u}$$

이를 간단한 코드로 다음과 같이 구현할 수 있습니다.

```python
import turtle
import math
import random

wn = turtle.Screen()
turtle.tracer(8,0)
alex = turtle.Turtle()
alex.hideturtle()
r = 200
for i in range(5000):
  u = random.random()
  d = r*(u**0.5)
  theta = random.random()*360
  x = d*math.cos(math.radians(theta))
  y = d*math.sin(math.radians(theta))
  alex.penup()
  alex.setposition(x,y)
  alex.dot()
turtle.update()
wn.mainloop()
```