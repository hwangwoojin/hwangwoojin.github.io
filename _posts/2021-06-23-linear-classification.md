---
layout: post
title: 선형분류
date:   2021-06-23 12:00:00 +0900
description: 선형분류에 대해서 알아보자
categories: 프로그래머스-인공지능-데브코스
---

**분류(Classification)**는 입력벡터 $$x$$를 $$K$$개의 가능한 클래스 중에서 하나의 클래스로 할당하는 문제입니다.

분류를 위한 결정이론으로는 확률을 사용할 수 있고, 사용하지 않을 수도 있습니다.

확률을 사용하지 않는 경우, 입력 $$x$$를 클래스로 할당하는 **판별함수(Discriminant function)**를 직접 찾게 됩니다.

확률을 사용하는 **확률적 모델(probabilistic model)**에는 사전확률과 우도를 모델링한 후 베이즈 확률을 사용해서 사후 확률을 구하는 생성모델(Generative model)과 우도를 직접적으로 모델링하는 식별모델(Discriminative model)이 존재합니다.

## 판별함수

**판별함수(Discriminant function)**은 따로 확률을 계산하지 않고 입력을 클래스로 할당하는 함수입니다. 선형분류 문제에서 사용되는 선형판별함수는 다음과 같은 형태를 가지고 있습니다.

$$y(x)=w^Tx+w_0$$

$$w$$는 가중치(weight) 벡터이고, $$w_0$$는 바이어스입니다.

이 선형판별함수는 $$y(x)\geq 0$$인 경우 이를 $$C_1$$으로, $$y(x)\leq 0$$인 경우를 $$C_2$$로 판별하게 됩니다. 이 때 어느 클래스로 분류할지를 정하는 **결정경계(Decision boundary)**는 $$y(x)=0$$이 됩니다. 선형분류문제에서 결정경계는 항상 선형적인 특징을 가집니다.

두개의 클래스가 아닌 다수의 클래스를 판별하는 선형판별함수는 다음과 같이 표현할 수 있습니다.

$$y_k(x)=w_k^Tx+w_{k0}, k=1,2,...,K$$

이 선형판별함수는 $$j\neq k$$일 때 $$y_k(x)>y_j(x)$$를 만족해야 입력벡터 $$x$$를 클래스 $$C_k$$로 판별합니다.

## 확률적 생성 모델

**확률적 생성 모델(Probabilistic generative models)**은 사전확률과 우도를 모델링하고 베이즈 확률을 사용해서 사후확률을 구하는 방법입니다. 위의 판별함수에서는 최적의 가중치 벡터를 찾는 것이 목적이라고 한다면, 확률적 모델에서는 데이터와 클래스의 분포를 모델링하면서 결과적으로 분류 문제를 풀게 됩니다.

두개의 클래스가 존재하는 경우, 사후 확률은 다음과 같이 구할 수 있습니다.

$$p(C_1\vert x)=\frac{p(x\vert C_1)p(C_1)}{p(x\vert C_1)p(C_1)+p(x\vert C_2)p(C_2)} = \frac{1}{1+exp(-a)}=\sigma(a)$$

$$a=\ln\frac{p(x\vert C_1)p(C_1)}{p(x\vert C_2)p(C_2)}$$

이 때 $$\sigma(a)$$를 **로지스틱 시그모이드 함수(Logistic sigmoid function)**라고 합니다.

다중 클래스 문제에서 사후 확률은 다음과 같이 구할 수 있습니다.

$$p(C_k\vert x)=\frac{p(x\vert C_k)p(C_k)}{\sum_jp(x\vert C_j)p(C_j)} = \frac{exp(a_k)}{\sum_jexp(a_j)}$$

$$a_k=p(x\vert C_k)p(C_k)$$

분류문제에서 두개의 클래스만 존재하는 경우 최대우도추정법을 통해 $$p(C_0), p(C_1), \mu_0,\mu_1$$을 구할 수 있는데, 정리하면 다음과 같은 값을 가지게 됩니다.

$$p(C_0)=\frac{1}{N}\sum_{n=1}^Nt_n=\frac{N_0}{N}=\frac{N_0}{N_0+N_1}$$

$$p(C_1)=\frac{1}{N}\sum_{n=1}^N(1-t_n)=\frac{N_1}{N}=\frac{N_1}{N_0+N_1}$$

$$\mu_0=\frac{1}{N_0}\sum_{n=1}^Nt_nx_n$$

$$\mu_1=\frac{1}{N_1}\sum_{n=1}^N(1-t_nx_n)$$

이때 $$N$$은 샘플의 수를 의미합니다.

## 코드

이제 파이썬 코드를 사용하여 분류 문제를 해결해보도록 하겠습니다. 다음 파이썬 라이브러리를 사용하였습니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns
```

`sklearn.datasets.make_classification`은 선형분류 문제를 위한 데이터를 만들어주는 함수입니다. 이 함수를 통해 간단한 데이터를 만들 수 있습니다. `features = 2`를 통해 클래스가 2개인 경우의 분류문제 데이터를 만들도록 하겠습니다.

```python
X, t = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

t = t[:,np.newaxis]

sns.set_style('white')
sns.scatterplot(X[:,0],X[:,1],hue=t.reshape(-1))
```

선형분류를 위한 시그모이드 함수와 에러함수는 다음과 같이 정의했습니다. 이 때 에러함수는 음의 로그우도를 사용하였습니다.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(X, t, w):
    N = len(t)
    h = sigmoid(X @ w)
    epsilon = 1e-5
    cost = (1/N)*(((-t).T @ np.log(h + epsilon))-((1-t).T @ np.log(1-h + epsilon)))
    return cost
```

배치 학습을 수행한 코드는 다음과 같습니다.

```python
def gradient_descent(X, t, w, learning_rate, iterations):
    N = len(t)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        w = w - (learning_rate/N) * (X.T @ (sigmoid(X @ w) - t))
        cost_history[i] = compute_cost(X, t, w)

    return (cost_history, w)

N = len(t)

X = np.hstack((np.ones((N,1)),X))
M = np.size(X,1)
w = np.zeros((M,1))

iterations = 1000
learning_rate = 0.01

initial_cost = compute_cost(X, t, w)

print("Initial Cost is: {} \n".format(initial_cost))
# Initial Cost is: [[0.69312718]] 

(cost_history, w_optimal) = gradient_descent(X, t, w, learning_rate, iterations)

print("Optimal Parameters are: \n", w_optimal, "\n")
# Optimal Parameters are: 
#   [[-0.07024012]
#   [ 1.9275589 ]
#   [ 0.02285894]] 

plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()
```

이 때 `iteration` 값은 전체 데이터를 읽어들이는 횟수를 의미하는 것으로, 데이터를 많이 읽으면 읽을수록 학습데이터에 대한 정확도는 높아지기 때문에 에러 함수에 대한 그래프의 값은 계속 줄어드는 양상을 보이게 됩니다.

이 선형분류 문제에서 정확도는 다음과 같이 구할 수 있습니다.

```python
def predict(X, w):
    return np.round(sigmoid(X @ w))

y_pred = predict(X, w_optimal)
score = float(sum(y_pred == t))/ float(len(t))

print(score)
# 0.954
```

이 때 결정경계는 다음과 같습니다. `coef`는 직선의 기울기를 말하며, `intercept`는 y절편을 말합니다.

```python
coef = -(w_optimal[1] / w_optimal[2])
# -53.93417067

intercept = -(w[0] / w_optimal[2])
# -0.
```

만약 전체 데이터를 한번에 학습시키는 배치 방식이 아니라 한번에 일정 배치만큼을 학습시키는 **미니 배치(mini batch)**방법을 사용하기 위해서는 다음과 같이 구현할 수도 있습니다.

```python
def batch_gd(X, t, w, learning_rate, iterations, batch_size):
    N = len(t)
    cost_history = np.zeros((iterations,1))
    shuffled_indices = np.random.permutation(N)
    X_shuffled = X[shuffled_indices]
    t_shuffled = t[shuffled_indices]

    for i in range(iterations):
        i = i % N
        X_batch = X_shuffled[i:i+batch_size]
        t_batch = t_shuffled[i:i+batch_size]
        if X_batch.shape[0] < batch_size:
            X_batch = np.vstack((X_batch, X_shuffled[:(batch_size - X_batch.shape[0])]))
            t_batch = np.vstack((t_batch, t_shuffled[:(batch_size - t_batch.shape[0])]))
        w = w - (learning_rate/batch_size) * (X_batch.T @ (sigmoid(X_batch @ w) - t_batch))
        cost_history[i] = compute_cost(X_batch, t_batch, w)

    return (cost_history, w)
```

이 미니 배치 방식은 `batch_size`를 통해 정해진 배치의 크기만큼의 데이터만 학습하는 방식입니다. 리스트에서 만약 남은 배치 크기가 부족하다면, 앞부분으로 채워주도록 구현하였습니다.

```python
X, t = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

t = t[:,np.newaxis]

N = len(t)

X = np.hstack((np.ones((N,1)),X))
M = np.size(X,1)
w = np.zeros((M,1))

iterations = 1000
learning_rate = 0.01

initial_cost = compute_cost(X, t, w)

print("Initial Cost is: {} \n".format(initial_cost))
# Initial Cost is: [[0.69312718]] 

(cost_history, w_optimal) = batch_gd(X, t, w, learning_rate, iterations, 32)

print("Optimal Parameters are: \n", w_optimal, "\n")
# Optimal Parameters are: 
#   [[-0.06442208]
#   [ 1.92701809]
#   [ 0.03572908]] 

plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()
```

```python
## Accuracy
y_pred = predict(X, w_optimal)
score = float(sum(y_pred == t))/ float(len(t))

print(score)
# 0.952
```

또는 **스토캐스틱 경사 하강법(SGD, Stochastic gradient descent)**을 사용할 수도 있습니다.

```python
def sgd(X, t, w, learning_rate, iterations):
    N = len(t)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        i = i % N
        w = w - learning_rate * (X[i, np.newaxis].T * (sigmoid(X[i] @ w) - t[i]))
        cost_history[i] = compute_cost(X[i], t[i], w)

    return (cost_history, w)
```

이 때 `iteration`은 위의 배치 학습과 미니 배치 학습에서와는 달리, 읽을 데이터의 갯수를 의미하게 됩니다. 따라서 이 SGD를 통해 학습한 그래프의 에러함수는 데이터의 갯수인 500개까지 들쭉날쭉하다가 그 이상부터는 일정하게 됩니다.

```python
X, t = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

t = t[:,np.newaxis]

N = len(t)

X = np.hstack((np.ones((N,1)),X))
M = np.size(X,1)
w = np.zeros((M,1))

iterations = 1000
learning_rate = 0.01

initial_cost = compute_cost(X, t, w)

print("Initial Cost is: {} \n".format(initial_cost))
# Initial Cost is: [[0.69312718]] 

(cost_history, w_optimal) = sgd(X, t, w, learning_rate, iterations)

print("Optimal Parameters are: \n", w_optimal, "\n")
# Optimal Parameters are: 
#   [[-0.19304782]
#   [ 2.5431236 ]
#   [ 0.01130098]] 

plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()
```

```python
## Accuracy
y_pred = predict(X, w_optimal)
score = float(sum(y_pred == t))/ float(len(t))

print(score)
# 0.96
```