---
layout: post
title: 기계학습 소개
date:   2021-06-23 15:00:00 +0900
description: 기계학습이란?
categories: 오일석-기계학습
---

> 오일석 교수님의 책 [기계 학습](http://www.yes24.com/Product/Goods/57537091)을 참고하였습니다.

## 기계학습이란?

**기계학습(Machine-Learning)**이란 주어진 작업에 대해서 경험을 통해 성능을 향상시키는 프로그램을 말합니다. 사람이 과거의 경험으로부터 배우는 것처럼, 기계학습은 과거의 데이터를 사용하여 성능을 최적화합니다.

전통적인 프로그래밍은 입력을 받아 알고리즘을 통해 결과를 출력하는 반면에, 기계학습은 입력과 목표값을 받아 이를 기반으로 스스로 학습한 후 결과를 출력하게 됩니다.

예를 들어 이동체의 시간에 따른 위치 데이터가 주어졌다고 하겠습니다. 그러면 임의의 시간이 주어질 때 이동체의 위치를 예측하는 문제를 생각해 볼 수 있습니다. 만약 목표치가 실수라면 **회귀(Regression)**문제이고, 목표치가 클래스라면 **분류(Classification)**문제가 됩니다.

## 훈련과 추론

**훈련집합(training set)**은 훈련에 사용되는 데이터 집합이자 특징값 $$X$$과 목표값 $$Y$$을 가지는 집합을 말합니다. 예를 들어 이동체 예시에서 $$X=\{x_1=2.0,x_2=4.0,x_3=6.0\}, Y=\{y_1=3.0,y_2=4.0,y_3=5.0\}$$와 같은 값이 훈련집합이 될 수 있습니다.

이 훈련집합의 데이터에서는 직관적으로 직선 형태의 데이터라는 가설을 세운 후 직선 모델을 사용하여 문제를 해결할 수 있습니다. 파라미터 $$w,b$$를 사용해서 다음과 같은 식을 세울 수 있습니다.

$$y=wx+b$$

**훈련(train)**은 주어진 문제에 대해 예측을 가장 정확하게 할 수 있는 최적의 매개변수를 찾는 작업입니다. 임의의 $$w,b$$를 설정한 후, 이를 계속 개선해서 최적화시키는 것이 목표입니다.

**추론(inference)**은 훈련을 시키지 않은 새로운 특징에 대해서 예측을 통해 목표값을 만들어내는 작업입니다. 예를 들어 입력값 $$x_4=8.0$$에 대해서 목표값을 $$y_4=6.0$$으로 예측하는 과정을 말합니다.

기계학습의 목적은 훈련을 통해 훈련집합에 속하지 않는 **테스트집합(test set)** 데이터에 대한 추론의 오류를 최소화하는 것입니다. 테스트집합에 대한 성능을 **일반화(Generalization)**라고 합니다. 예를 들어 모의고사와 수능을 모두 잘 본 학생은 일반화 성능이 높다고 할 수 있지만, 모의고사만 잘 보고 수능을 잘 보지 못한 학생은 일반화 성능이 낮다고 말할 수 있습니다.

## 특징공간

**특정공간**이란 데이터가 표현되는 공간을 말합니다. 이동체의 예시와 같이 시간에 대해서 이동값이 주어지는 경우, 특징은 이동값이므로 이 데이터는 1차원 특징공간을 가지게 됩니다. 모 고등학교의 몸무게와 키 데이터는 2개의 특징값을 가지므로 2차원 특징공간을 가지게 됩니다.

이외에도 꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비를 특징으로 가지는 Iris 데이터는 4차원 특징공간을 가지고, 28x28(784) 크기의 흑백 이미지를 가지는 MNIST 손글씨 숫자 데이터는 784차원의 특징공간을 가지게 됩니다.

d-차원 데이터를 직선모델을 사용한다고 할 떄 필요한 매개변수 수는 $$d+1$$이 됩니다.

$$y=w_1x_1+w_2x_2+...+w_dx_d+b$$

만약 2차 곡선 모델을 사용한다고 하면 매개변수수는 지수적으로 증가하며, $$d^2+d+1$$가 됩니다.

$$y=w_1x_1^2+w_2x_2^2+...+2_dx_{d^2}+w_{d+1}x_1x_2+...+w_{d^2}x_{d-1}x_d+w_{d^2+1}x_1+...+w_{d^2+d}x_d+b$$

모델의 차수를 늘릴수록 필요한 매개변수의 수는 기하급수적으로 증가하게 됩니다.

## 차원의 저주

**차원의 저주(Curse of dimensionality)**란 차원이 커지면서 매개변수의 수는 기하급수적으로 커지는데, 이에 비해 데이터는 매우 희소해서 유의미한 특징을 찾지 못하는 문제입니다.

예를 들어 784차원을 가지는 MNIST 데이터의 특징공간은 $$2^{784}$$의 크기를 가집니다. 그러나 이에 비해 샘플은 고작 6만개 정도밖에 되지 않습니다.

이를 해결하기 위해서 두가지를 가정합니다. 하나는 **데이터 희소(Data sparsity)** 특성 가정입니다. 이는 특징공간의 대부분이 현실적으로 생성될 가능성이 희박한 특징공간이라는 것을 의미합니다.

두번째는 **매니폴드 가정(Manifold assumption)**입니다. 고차원의 데이터는 낮은 차원의 매니폴드에 가깝게 집중되어 있다는 것을 의미합니다. 매니폴드 위의 데이터는 일정한 규칙에 따라 매끄럽게 변화하게 됩니다.

## 선형 분리 불가능

데이터가 선형 분리 불가능할 경우, 선형모델을 사용해서 해결할 수 없는 문제가 존재합니다. 예를 들어 XOR 문제는 선형으로 분리할 수 없습니다.

이를 해결하기 위해서는 공간 변환을 통해 데이터를 선형 가능한 특징 공간으로 변환시킬 수 있습니다.

## 표현 학습과 심층 학습

**표현 학습(Representation learning)**이란 주어진 데이터를 분석하여 좋은 특징 공간을 찾는 작업입니다. 선형 분리 불가능한 특징공간을 선형 분리 가능하도록 변환하거나, 직교좌표계를 극좌표계로 변환하여 선형 분리 가능하도록 하는 방법입니다.

이 외에도 사진으로부터 특징적인 표현을 학습하는 것 또한 표현학습이라고 할 수 있습니다. 개구리 사진으로부터 눈, 다리 등의 특징을 찾을 수 있도록 합니다.

다수의 은닉층을 가진 신경망을 활용하여 표현학습을 수행하는 것을 **심층학습(Deep learning)**이라고 합니다. 심층학습의 은닉층은 앞부분에서는 선, 점 등의 저급 특징을 학습하며, 뒷부분에서는 얼굴 등의 추상화된 특징을 추출하는 성질이 있습니다.

## 기계학습과 데이터

현실의 데이터들은 어떻게 생성되었는지 그 과정을 알 수 없는 경우가 대부분입니다. 따라서 주어진 훈련집합 $$X,Y$$를 통해 훈련한 가설 모델을 통한 근사 추정을 통해 해결합니다.

데이터는 주어진 과업에 적합하도록 수집해야 합니다. 데이터가 많을수록 과업의 성능은 일반적으로 향상합니다. 단 데이터를 다양하게 수집해야 합니다. 예를 들어 데이터를 정면 얼굴만 수집한다면, 측면 얽굴에 대해서는 매우 낮은 성능을 보이게 됩니다.

## 목적함수

**목적함수(Object function)** 또는 비용함수(Cost function)란 과업에서 학습을 위해 설정한 정량적 기준이자 성능측정의 판단기준을 말합니다. 예를 들어 예측함수 $$f_\theta$$와 실제 목표값 $$y$$의 차이의 제곱합인 평균제곱오차(MSE, Mean Squared Error)는 목적함수로 사용할 수 있습니다.

$$J(\theta)=\frac{1}{n}\sum_{i=1}^n(f_\theta(x_i)-y_i)^2$$

이 때 $$f_\theta(x_i)-y_i$$를 오차(error) 또는 손실(loss)이라고 합니다.

학습을 수행할 때 처음에는 $$w,b$$를 임의의 난수로 설정한 후, 이를 목적함수를 최소화하는 방향으로 계속 개선하게 됩니다.

$$\hat\theta=argmin_\theta J(\theta)$$

알고리즘 형식으로는 다음과 같습니다. 이 때 미분을 사용하여 기울기가 감소하는 방향으로 학습이 일어나게 됩니다.

```python
난수를 생성하여 초기 해 a를 설정한다.
while (J(a)가 0.0 에 충분히 가깝지 않음):
    d = (J(a)가 작아지는 방향)
    a = a + d
```

## 과소적합과 과잉적합

**과소적합(under-fitting)**은 모델의 용량이 작아서 오차가 큰 현상을 말합니다. 예를 들어 1차, 2차 등 저차원 모델을 사용한 경우 발생할 수 있습니다. 이는 비선형 모델이나, 차원이 큰 모델을 사용하면 오차를 줄일 수 있습니다.

**과잉적함(over-fitting)**은 모델의 용량이 너무 커서 학습과정에서 노이즈까지 학습하는 경우를 말합니다. 이 경우 훈련집합에 대해서는 완벽하게 근사하지만, 새로운 테스트집합에 대해서는 큰 오차를 보이게 됩니다. 왜냐하면 훈련집합을 단순 암기했기 때문입니다. 이를 위해서는 적절한 용량의 모델을 선택해야 합니다.

현실에서는 경험적으로 큰 틀의 모델을 선택한 후, 세부적으로 작은 모델을 선택하는 전략을 취합니다. 또 모델이 정상을 벗어나지 않도록 여러 규제 기법을 적용합니다.

## 편향과 분산

**편향(bias)**은 전체 데이터가 목표값으로부터 떨어져있는 정도를 말합니다. 편향이 크다는 것의 의미는 모든 데이터가 전체적으로 목표값과 떨어져있다는 의미입니다. 보통 적은 차원의 모델을 사용했을 때, 편향이 크게 됩니다.

**분산(variance)**은 데이터간의 떨어져있는 정도를 말합니다. 분산이 크다는 것은 데이터와 데이터 사이의 거리가 멀다는 것을 의미하며, 분산이 작다는 것은 데이터가 대체로 뭉쳐있다는 것을 의미합니다. 보통 높은 차원의 모델을 사용했을 때 분산이 크게 됩니다.

편향과 분산은 서로 trade-off 관계인 경우가 많습니다. 적은 차원의 모델을 사용하면 편향은 크고, 분산은 작아 항상 비슷한 모델을 얻습니다. 반면에 높은 차원의 모델을 사용하면 편향은 적으나 분산이 크게 되어 항상 다른 모델을 얻습니다.

기계학습의 목표는 편향과 분산 모두가 낮은 예측 모델을 만드는 것입니다. 따라서 이를 적절히 설정해야 합니다.

## 검증집합

과소적합과 과잉적합 문제를 해결하여 일반화 성능을 높이기 위해서 **검증집합(Validation set)**을 사용하는 경우가 많습니다. 검증집합이란 훈련집합, 테스트집합과는 다른 별도의 집합을 말합니다. 훈련집합으로 학습한 모델을 검증집합으로 성능을 측정하여, 모델을 결정하게 됩니다.

```python
for (각각의 모델):
    모델을 훈련집합으로 학습시킨다.
    검증집합을 이용해서 학습된 모델의 성능을 측정한다.
가장 높은 성능을 보인 모델을 선택한다.
테스트 집합으로 선택된 모델의 성능을 측정한다.
```

**교차검증(Cross validation)**은 비용문제 등으로 인해 따로 검증집합이 없는 경우 사용되는 방식입니다. 훈련집합의 일부를 분리해서 이를 검증집합으로 두고 검증하는 방식입니다. 이를 여러번 반복하기도 하는데, 예를 들어 5겹 교차 검증의 경우 훈련집합을 5등분하여 그중 하나씩 돌아가며 검증집합으로 두어 성능을 측정합니다.

```python
훈련집합을 k개로 등분한다.
for (각각의 모델):
    for (i = 1...k):
        i번째 그룹을 제외한 k-1 개 그룹으로 모델을 학습시킨다.
        학습된 모델의, 성능을 i번째 그룹으로 측정한다.
    k개 성능을 평균하여 해당 모델의 성능으로 취한다.
가장 높은 성능을 보인 모델을 선택한다.
테스트 집합으로 선택된 모델의 성능을 측정한다.
```

**부트스트랩(bootstrap)**은 데이터 분포가 불균형일 때 주로 사용하는 방법입니다. 훈련집합에서 특정 갯수만큼을 샘플링해서 새로운 훈련집합으로 구성하고, 나머지로 검증집합을 구성하여 훈련시키는 방식입니다.

```python
for (각각의 모델):
    for (i = 1...k):
        훈련집합에서 중복을 허용하는 n개의 샘플을 뽑아 새로운 훈련집합을 구성한다.
        새로운 훈련집합으로 모델을 학습시킨다.
        나머지 집합으로 모델의 성능을 측정한다.
    K개 성능을 평균하여 해당 모델의 성능으로 취한다.
가장 높은 성능을 보인 모델을 선택한다.
테스트 집합으로 선택된 모델의 성능을 측정한다.
```

## 데이터 확대

데이터를 모으는 일은 매우 많은 비용이 드는 문제입니다. 왜냐하면 사람이 일일이 라벨링 작업을 해야 하기 때문입니다. 따라서 인위적으로 데이터를 회전시키거나 왜곡시켜서 데이터를 늘리는 방법을 사용합니다. 이를 **데이터 확대(Data augmentation)**라고 합니다.

## 규제

**가중치 감쇠(Weight decay)**는 모델 파라미터의 가중치를 작게 조절하는 규제 기법입니다. 고차원 모델을 사용하는 경우, 파라미터의 가중치가 매우 크게 나타나며 과적합 현상이 일어나는 경우가 존재합니다. 이 경우 가중치 감쇠가 적용된 목적함수를 사용해서 가중치를 작게 조절할 수 있습니다. 예를 들어 규제항 $$\lambda\vert\vert\theta\vert\vert_2^2$$을 기존 목적함수에 더하는 것으로 다음과 같이 규제를 적용할 수 있습니다.

$$J(\theta)=\frac{1}{n}\sum_{i=1}^n(J_\theta(x_i)-y_i)^2+\lambda\vert\vert\theta\vert\vert_2^2$$

이 외에도 가중치 벌칙, 조기 멈춤, 드롭아웃, 앙상블 등의 규제가 존재하며 이에 대해서는 이후 글에서 다루도록 하겠습니다.
