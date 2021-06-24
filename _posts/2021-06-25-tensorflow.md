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
