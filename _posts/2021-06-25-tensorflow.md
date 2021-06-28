---
layout: post
title: Tensorflow
date:   2021-06-25 04:00:00 +0900
description: Tensorflow 의 기본적인 사용법에 대해 알아보자
categories: 프로그래머스-인공지능-데브코스
---

텐서플로우는 구글에서 만든 머신러닝을 위한 라이브러리입니다. 텐서플로우에 대한 더 많은 정보는 [tensorflow 공식문서](https://tensorflow.org/guide/tensor)에서 확인할 수 있습니다.

## tensorflow 모듈 가져오기

코드는 구글 코랩 환경에서 수행하였습니다.

```python
import tensorflow as tf

tf.__version__
# '2.5.0'
```

## 초기화

텐서플로우에서는 기본적으로 `tf.Variable`을 통해 변수를 초기화할 수 있습니다.

```python
x = tf.Variable(tf.zeros([1,2,3]))
x
# <tf.Variable 'Variable:0' shape=(1, 2, 3) dtype=float32, numpy=array([[[0., 0., 0.],[0., 0., 0.]]], dtype=float32)>
```

크기가 아니라 직접 값을 통해 초기화할 수도 있습니다.

```python
x = tf.Variable(0,0)
x
# <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=0>
```

다른 행렬을 기준으로 해서 새로운 행렬을 생성할 수도 있습니다. 이 경우 간단하게 덧셈 기호를 사용하면 됩니다.

```python
y = x + 1
x
# <tf.Tensor: shape=(), dtype=int32, numpy=1>
```

만약 텐서플로우 변수에 특정 값을 할당하고 싶다면 `tf.assign_add` 함수를 사용하면 됩니다. 추가한 값을 읽어들이기 위해 `tf.read_value` 함수를 사용하였습니다.

```python
x = tf.Variable(0,0)
x.assign_add(1) # 1 값을 추가
x.read_value()
# <tf.Tensor: shape=(), dtype=int32, numpy=1>
```

## 랭크

텐서플로우의 변수는 모두 랭크(rank)를 가집니다. 랭크란 행렬의 차원을 말하는 개념입니다. 랭크가 0이면 값 하나를 가지고 있는 것이고, 랭크가 1이면 벡터, 랭크가 2이면 행과 열을 가진 행렬이 됩니다. 랭크는 `tf.rank` 함수를 통해 구할 수 있습니다.

```python
x = tf.Variable([[4],[9],[16],[25]],tf.int32)
tf.rank(x)
# <tf.Tensor: shape=(), dtype=int32, numpy=2>
```

예를 들어 배치 크기, 높이, 너비, 색상값을 가지는 이미지에 대한 행렬이 존재한다고 할 떄, 이 행렬의 차원은 4가 됩니다.

```python
image = tf.zeros([10, 128, 128, 3])
tf.rank(image)
# <tf.Tensor: shape=(), dtype=int32, numpy=4>
```

## 원소 참조

텐서플로우에서 원소를 참조하는 방법은 파이썬의 리스트와 같습니다.

## 차원 변환

텐서플로우에서 차원 변환은 `tf.reshape` 함수를 통해 사용됩니다. 이는 torch에서 `view` 함수와 개념적으로 매우 유사합니다.

```python
x = tf.ones([3,4,5])
x.shape
# TensorShape([3, 4, 5])
```

```python
y = tf.reshape(x, [6,10]) # 6*10
y.shape
# TensorShape([6, 10])
```

만약 -1로 설정하는 경우, 차원을 자동으로 계산하여 적절한 값으로 설정합니다. 예를 들어 6x10 행렬을 3x-1 행렬로 변환한다고 하면 자동으로 3x20 행렬로 변환되게 됩니다.

```python
z = tf.reshape(y, [3,-1])
z.shape
# TensorShape([3, 20])
```

## 자료형 변환

자료형을 변환하기 위해서는 `tf.cast` 함수를 사용합니다. 이 떄 변환하고자 하는 자료형을 `dtype` 파라미터에 넣어주면 됩니다.

```python
x = tf.constant([1,2,3])
x
# <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>

```

위 코드에서 행렬이 int32 자료형으로 저장된 것을 확인할 수 있습니다. 이를 실수 자료형인 float32 자료형으로 전환하는 코드는 다음과 같습니다.

```python
y = tf.cast(x, dtype=tf.float32)
y
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>
```

## 미분

`tensorflow`에서는 `tf.GradientTape`를 통해서 자동미분을 수행할 수 있습니다. 그레디언트 테이프는 실행된 모든 연산을 테이프에 기록해서 이후 미분을 수행하는 방식입니다.

아래 코드와 같이 그레디언트 테이프를 사용하여 미분을 구할 수 있습니다.

```python
x = tf.ones((2, 2))
# [[1, 1],
#  [1, 1]]

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  print('y: ', y)
  # y:  tf.Tensor(4.0, shape=(), dtype=float32)

  z = tf.multiply(y, y)
  print('z: ', z)
  # z:  tf.Tensor(16.0, shape=(), dtype=float32)

# 입력 텐서 x에 대한 z의 도함수
dz_dx = t.gradient(z, x)
print(dz_dx)
# tf.Tensor(
# [[8. 8.]
#  [8. 8.]], shape=(2, 2), dtype=float32)
```

입력 텐서 x에 대한 y의 도함수도 구할 수 있습니다.

```python
with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  print('y: ', y)
  # y:  tf.Tensor(4.0, shape=(), dtype=float32)

  z = tf.multiply(y, y)
  print('z: ', z)
  # z:  tf.Tensor(16.0, shape=(), dtype=float32)

# 입력 텐서 x에 대한 y의 도함수
dz_dy = t.gradient(z, y)
print(dz_dy)
# tf.Tensor(8.0, shape=(), dtype=float32)
```

이 때, `t.gradient`를 두번 호출하게 된다면 에러가 발생하게 됩니다. 왜냐하면 `t.gradient`를 호출할 때 그레디언트 테이프의 리소스가 해제되기 때문입니다. 만약 여러번 호출하고 싶다면 그레디언트 테이프에 `persistent=True` 옵션을 설정하면 됩니다. 이 경우 `del`을 통해 직접 리소스를 해제해야 합니다.

```python
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = x * x
  z = y * y  # z = x ^ 4
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
print(dz_dx)
dy_dx = t.gradient(y, x)  # 6.0   (2 * x at x = 3)
print(dy_dx)
del t  # 테이프에 대한 참조를 삭제합니다.
```

이를 응용하면 다음과 같이 그레디언트 함수를 설정할 수도 있습니다.

```python
def f(x, y):
  output = 1.0
  for i in range(y):
    if i > 1 and i < 5:   # output(1) * 2 * 3 * 4
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

print(grad(x, 6).numpy())
# 12.0
```

그레디언트 테이프 안에 그레디언트 테이프를 선언할 수 있습니다. 이는 높은 차수의 그레디언트를 구할때 사용합니다. 예를 들어 다음과 같이 2차 그레디언트를 구할 수 있습니다.

```python
x = tf.Variable(1.0) 

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x
  # 't' 컨텍스트 매니저 안의 그래디언트를 계산합니다.
  # 이것은 또한 그래디언트 연산 자체도 미분가능하다는 것을 의미합니다. 
  dy_dx = t2.gradient(y, x)      # dy_dx = 3 * x^2 at x = 1
d2y_dx2 = t.gradient(dy_dx, x)   # d2y_dx2 = 6 * x at x = 1

print(dy_dx.numpy())
# 3.0

print(d2y_dx2.numpy())
# 6.0
```

## 신경망 만들기

`tensorflow`에서 신경망은 다음과 같이 만들 수 있습니다. 이 때 `layers.Dense`는 `torch`에서 `nn.Linear`와 일치합니다.

```python
import tensorflow as tf
from tensorflow.keras import layers

layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Conv1d(4, name="layer3")

x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
y.shape
# TensorShape([3, 4])
```

혹은 `keras.Sequential`을 사용하여 다음과 같이 정의할 수도 있습니다.

```python
from tensorflow import keras

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),  # Pytorch - nn.Linear 
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)

x = tf.ones((3, 3))
y = model(x)
model.layers
# [<tensorflow.python.keras.layers.core.Dense at 0x7f0750baa6d0>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7f075af135d0>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7f075af12fd0>]
```

`add`를 통해 하나씩 층을 추가할 수도 있습니다.

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
model.layers
# [<tensorflow.python.keras.layers.core.Dense at 0x7f0750baa6d0>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7f075af135d0>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7f075af12fd0>]
```

## Fashion MNIST 데이터 학습하기

`Fashion MNIST`는 패션에 관한 10종류의 흑백 이미지를 가지고 있는 데이터셋입니다. `tensorflow`를 사용하여 이 데이터셋에 대한 학습을 수행해보도록 하겠습니다.

```python
import tensorflow as tf
import numpy as np

from tensorflow import keras

# 데이터셋 불러오기
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 데이터 전처리
# 데이터의 RGB 값을 0에서 1 사이의 값으로 조정하기
train_images = train_images / 255.0
test_images = test_images / 255.0

# 신경망 모델 구성
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 모델의 정보를 확인하고 싶다면,
# model.summary()

# 손실 함수와 옵티마이저 정하기
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습 수행
model.fit(train_images, train_labels, epochs=5)

# 학습 데이터에 대한 정확도 구하기
test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=2)

print(test_loss)
# 0.5950284600257874

print(test_accuracy)
# 0.8130999803543091
```
