
# coding: utf-8

# # Full implementation using the CIFAR10 dataset.

# In[1]:


# Custom utility class
from utils import *

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

# Special package provided by pytorch
import torchvision
import torchvision.transforms as transforms


# In[2]:


## Hyperparameters.

# If CUDA is enabled, use the GPU, otherwise use the CPU.
has_gpu = torch.cuda.is_available()

# image channel 3=RGB, 1=Grayscale
img_channels = 3

# Class labels.
classes = CLASS_LABELS['cifar']
num_classes = len(classes)

# Data directory.
data_dir = '../datasets/cifar'  # Dataset directory.
download = True                 # Download dataset iff not already downloaded.
normalize = 0.5                 # Normalize dataset.

# Training parameters
batch_size = 16  # Mini-batch size.
lr = 1e-2        # Optimizer's learning rate.
epochs = 5       # Number of full passes over entire dataset.


# In[3]:


# Should normalize images or not.
# Normalization helps convergence.
if normalize:
    # Transform rule: Convert to Tensor, Normalize images in range -1 to 1.
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), 
                                                         (0.5, 0.5, 0.5))])
else:
    # Transform rule: Convert to Tensor without normalizing image
    transform = transforms.Compose([transforms.ToTensor()])

# Download the training set and apply the transform rule to each.
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=download, transform=transform)
# Load the training set into mini-batches and shuffle them
trainset = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                       shuffle=True, num_workers=2)

# Download the testing set and apply the transform rule to each.
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=download, transform=transform)
# Load the testing set into mini-batches without shuffling.
testset = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                      shuffle=False, num_workers=2)


# In[4]:


# Let's visualize some training set.
visualize(trainset, smooth=True, classes=classes)


# In[5]:


class Network(nn.Module):
    
    def __init__(self, **kwargs):
        super(Network, self).__init__()
        
        # Hyper-parameters
        self._img_channels = kwargs.get('img_channels')
        self._num_classes = kwargs.get('num_classes')
        
        # 2 convolutional & 3 fully connected layers
        self.conv1 = nn.Conv2d(self._img_channels, 16, 2)
        self.conv2 = nn.Conv2d(16, 32, 2)
        flatten_size = self.conv2.out_channels * 7 * 7
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self._num_classes)
    
    def forward(self, x):
        # Convolutional layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        # Flatten layer
        x = x.view(-1, self.__flatten(x))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))  # relu + linear
        x = F.dropout(x, p=0.2)  # 20% dropout
        x = F.relu(self.fc2(x))  # relu + linear
        
        # Output layer
        x = self.fc3(x)  # linear
        
        return x
    
    def __flatten(self, x):
        size = x.size()[1:]  # Input shape excluding batch dim.
        return torch.Tensor(size).numel()


# In[6]:


# Instantiate the network and pass in our parameters.
net = Network(img_channels=img_channels, num_classes=len(classes))

# Make use of GPU if it's available.
if has_gpu:
    net = net.cuda()


# In[7]:


# Loss function criterion.
criterion = nn.CrossEntropyLoss()

# Adam optimizer.
optimizer = optim.Adam(net.parameters(), lr=lr)


# In[8]:


# Loop over the data multiple times.
for epoch in range(epochs):

    # Loop through the training dataset (batch by batch).
    for i, data in enumerate(trainset):
        
        # Get the inputs and labels.
        inputs, labels = data
        
        # Wrap them in Variable (explained in section 2),
        # and optionaly do that on the GPU.
        if has_gpu:
            # Use GPU.
            inputs, labels = Variable(inputs.cuda()), Variable(inputs.cuda())
        else:
            # Use CPU.
            inputs, labels = Variable(inputs), Variable(labels)
        
        # Zero the optimizer gradient buffer to prevent gradient accumulation.
        optimizer.zero_grad()
        
        # Forward and backward propagation.
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Update learnable parameters w.r.t the loss.
        optimizer.step()
        
        # Print statistics.
        print(f'\rEpoch: {epoch+1:,}\t Batch: {i+1:,}\t Loss: {loss.data[0]:.4f}', end='')

    # Line break.
    print()


print('\nFinished training!')


# In[9]:


# Visualize the testset, and it's prediction and,
# shows accuracy for each classes in the dataset.
accuracy(net=net, data=testset, classes=classes)

