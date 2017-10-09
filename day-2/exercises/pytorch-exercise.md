PyTorch Exercise
===============

Goals
-----

- Basic introduction to convnets
- First steps with PyTorch

Requirements
-----
- numpy
- matplotlib
- torch
- torchvision

The latter two can be installed through Anaconda:
```
conda install pytorch torchvision -c soumith
```


Exercise
-----

In this exercise, we follow one of the PyTorch tutorials and train a very simple network to recognize objects in 32x32 pixel images.
Because this involves lots of framework functions, it will be more of a "guided copy and paste" than actual programming.

### Task 1

Lets start with some imports that we will need:

```python
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
```

We want to train and test on the cifar10 dataset. Torch already contains code for (down)loading these datasets. For most learning approaches, it is beneficial to scale and shift the input data such that the mean is zero and the variance is one. This can be done inside the loader. Since the cifar data is scaled between 0 and 1 we assume a mean of 0.5 and a standard deviation of 0.5. This will let the loader scale it into the range of -1 and 1.
```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```
With this transform at hand, load the training data. Upon first call, this will download the data from the internet.
```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
```
Implement a second loader for the test dataset.

### Task 2

The following code fetches a mini batch of training data (4 samples):
```python
dataiter = iter(trainloader)
images, labels = dataiter.next()
```

Display the four classes. The names can be read from this array:
```python
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

Display the corresponding four images using:
```python
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
imshow(torchvision.utils.make_grid(images))
```

### Task 3

The following code builds a network with one convolution (operating on 3 channels, with 6 filters of size 5x5) + relu + pooling, a flattening operation, and a fully connected layer:
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 10)

    def forward(self, x):
        # x is 32x32x3
        x = self.pool(F.relu(self.conv1(x)))
        # x is 14x14x6 (Filters of size 5 without padding take away 4 pixels. Max pooling halves resolution.)
        x = x.view(-1, 6 * 14 * 14)
        x = self.fc1(x)
        return x

net = Net()
print(net)
```
The last layer does not have a relu activation and has one output for each class. Extend the network to 2x conv + relu + pool and 3x fully connected layers. Remember that, except for the last layer, the fully connected layers also need relu activations.

### Task 4

Train the network on the training data for 2 epochs using cross entropy loss and stochastic gradient descend:
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

### Task 5

Load a minibatch of "images" and "labels" from the test dataset.
Run the trained network on the minibatch of test data:
```python
outputs = net(Variable(images))

_, predicted = torch.max(outputs.data, 1)

imshow(torchvision.utils.make_grid(images))
```

Look at the images and compare the predicted labels "predicted" to the ground truth labels "labels".


### Task 6

Compute the accuracy on the entire test dataset:
```python
correct = 0
total = 0
for data in testloader:
    images, labels = data
    # run through network and update number of total and correct labels.

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

