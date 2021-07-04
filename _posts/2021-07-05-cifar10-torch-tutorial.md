---
layout: post
title: CIFAR10 pytorch 튜토리얼
date:   2021-07-05 07:00:00 +0900
description: pytorch를 사용하여 CIFAR10 데이터셋을 훈련해보자.
categories: 프로그래머스-인공지능-데브코스
---

[pytorch 공식문서](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)를 참고하여 작성하였습니다.

**CIFAR10** 데이터셋은 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭으로 총 10개의 클래스를 가지고 있으며 크기는 32x32인 RGB 데이터로 구성되어 있습니다.

여기서는 CIFAR10 이미지 데이터셋을 훈련하기 위해 다음 과정을 수행하도록 하겠습니다.

1. CIFAR10 데이터셋을 가져와서 정규화하기

2. CNN 모델 만들기

3. 손실 함수 정의하기

4. 훈련 데이터를 사용하여 신경망 훈련하기

5. 테스트 데이터를 사용하여 신경망 검사하기

## CIFAR10 데이터셋을 가져와서 정규화하기

여기서는 `torchvision`을 사용하여 간단하게 데이터를 가져와서 정규화할 수 있습니다.

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

만약 데이터셋의 이미지를 보고 싶다면 `matplotlib`을 사용할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
```

## CNN 모델 만들기

CNN 모델은 다음과 같이 만들 수 있습니다. 아래 모델은 3개의 fc 레이어를 가진 간단한 신경망 예시입니다.

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

## 손실 함수 정의하기

여기서는 `CrossEntropyLoss` 손실함수와 `SGD` 옵티마이저를 사용하도록 하겠습니다.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

## 훈련 데이터를 사용하여 신경망 훈련하기

데이터셋을 가져오고 모델, 손실함수, 옵티마이저를 정의했다면 훈련 데이터를 사용하여 모델을 훈련시킬 수 있습니다. 다음은 2번의 epoch을 통해 훈련하는 예시입니다.

```python
epochs = 2

for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
```

출력

```
[1,  2000] loss: 2.181
[1,  4000] loss: 1.852
[1,  6000] loss: 1.686
[1,  8000] loss: 1.599
[1, 10000] loss: 1.524
[1, 12000] loss: 1.484
[2,  2000] loss: 1.415
[2,  4000] loss: 1.376
[2,  6000] loss: 1.354
[2,  8000] loss: 1.329
[2, 10000] loss: 1.306
[2, 12000] loss: 1.293
```

훈련한 모델을 저장하는 코드는 다음과 같습니다.

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

## 테스트 데이터를 사용하여 신경망 검사하기

훈련 데이터를 통한 훈련이 끝났다면, 훈련된 모델을 테스트 데이터를 사용하여 검증해볼 수 있습니다. 다음은 테스트 데이터셋의 몇가지 이미지를 보여주는 코드입니다.

```python
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

출력

```
GroundTruth:    cat  ship  ship plane
```

이제 훈련한 모델을 가져와서 이를 사용하여 테스트 데이터셋을 검증해보고 그 정확도를 검사해볼 수 있습니다. 앞서 본 4개의 이미지를 예측해보도록 하겠습니다.

```python
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
```

출력

```
Predicted:   frog   car   car plane
```

전체 테스트 데이터셋에 대한 정확도는 다음과 같이 구할 수 있습니다.

```python
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

출력

```
Accuracy of the network on the 10000 test images: 55 %
```

약 55%의 정확도를 얻은 것을 확인할 수 있습니다. 클래스별로 정확도를 구하기 위한 코드는 다음과 같습니다.

```python
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
```

출력

```
Accuracy for class plane is: 57.4 %
Accuracy for class car   is: 78.3 %
Accuracy for class bird  is: 26.4 %
Accuracy for class cat   is: 37.1 %
Accuracy for class deer  is: 44.3 %
Accuracy for class dog   is: 43.0 %
Accuracy for class frog  is: 78.9 %
Accuracy for class horse is: 57.5 %
Accuracy for class ship  is: 65.6 %
Accuracy for class truck is: 70.2 %
```

가장 높은 정확도를 보인 것은 개구리(78.9%)이고 가장 낮은 정확도를 보인 것은 새(26.4%)인 것을 확인할 수 있습니다.

## CPU가 아닌 GPU로 학습하기

만약 모델을 CPU로 학습하고 싶다면, 위의 코드에서 모델, 입력값, 라벨값만 `.to(device)` 또는 `.cuda()` 를 사용해주면 됩니다.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
...
net.to(device)
...
inputs, labels = data[0].to(device), data[1].to(device)
```

여기까지 CIFAR10 데이터셋과 pytorch를 사용하여 간단한 학습을 수행해 보았습니다
