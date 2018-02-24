## Neural Networks

Neural networks can be constructed using the `torch.nn` package.

Now that you had a glipse of `autograd`, `nn` depends on `autograd` to define models and differentiate them. An `nn.Module` contains layers, and a method `forward(input)` that returns the `output`.

For example, look at a simple convolutional neural network that classifies digit images:

![Convnet](../images/mnist-convnet.png)

It is a simple convolutional neural network. It takes the input, feeds it through several layers one after the other and then finally gives the output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss *(how far is the output from being correct)*
- Propagate gradients back into the network's parameters
- Update the weights of the network, typically using a simple update rule: `weight = weight - learning_rate * gradient`

## Define the Network

Let's define this network:

In [1]:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
```

### Constructing the `Net` class

In [2]:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_channels: 1
        # out_channels: 6
        # kernel_size: 5x5 
        # Defaults->stride:1, padding:0, dialation:1, groups:1, bias:True
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # dense(or fully connected, or affine) layer
        # y = Wx + b
        # in_features: size of each input sample
        # out_features: size of each output sample
        # bias: Learn an additive bias. Default: `True`
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # Max pooling over a (2x2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten
        x = x.view(-1, self._flatten(x))
        # Apply relu to the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # The output layer does not need ReLU activation
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
    def _flatten(self, x):
        """
        Retrieve the flattened dimension excluding 
        the batch dimension.        
        
        :param x:torch.Tensor
            The tensor to extract it's flattened dim.        
        
        :return num_features: int
            Number of flattened dimension (except the batch dim).
        """
        # get all dimensions except for the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


```

In [3]:

```python
# Create an instance of the Net class
net = Net()
print(net)
```

```sh
Net(
  (conv1): Conv2d (1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d (6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120)
  (fc2): Linear(in_features=120, out_features=84)
  (fc3): Linear(in_features=84, out_features=10)
)

```

You have to define the `forward` function and the `backward` function *(where gradients are computed)* is automatically defined for you saying `autograd`. You can also use the Tensor opeartions in the `forward` function.

The learnable parameters of a model is returned by `net.parameters`

**NOTE:** `net.parameters()` returns a generator, therefore you might want to convert it to a *regular Python list*. As show here:

In [4]:

```python
params = list(net.parameters())
conv1_params = params[0]  # 1st layer (conv1) parameters
print(conv1_params.size())
```

```sh
torch.Size([6, 1, 5, 5])

```

In [5]:

```python
# Weights and bias are stored in `net.Module.weight` & `net.Module.bias`
# e.g.
conv1_weights = net.conv1.weight
conv2_bias = net.conv2.bias

fc1_weights = net.fc1.weight
fc2_bias = net.fc2.bias

print(f'conv1_weights.size() = {conv1_weights.size()}')
print(f'conv2_bias.size()    = {conv2_bias.size()}')

print(f'fc1_weights.size()   = {fc1_weights.size()}')
print(f'fc2_bias.size()      = {fc2_bias.size()}')
```

```sh
conv1_weights.size() = torch.Size([6, 1, 5, 5])
conv2_bias.size()    = torch.Size([16])
fc1_weights.size()   = torch.Size([120, 400])
fc2_bias.size()      = torch.Size([84])

```

The input to the `forward` is a `autograd.Variable`, and so is the output. **NOTE:** Expected input size to this net(LeNet) is 32x32. To use this net on MNIST, please resize the images from dataset to 32x32.

In [6]:

```python
X_input = Variable(torch.randn(1, 1, 32, 32))
out = net(X_input)
print(out)
```

```sh
Variable containing:
 0.0953  0.1049  0.1029  0.0885  0.1068  0.1005  0.1015  0.1115  0.0955  0.0926
[torch.FloatTensor of size 1x10]


```

Zero the gradient buffers of all parameters and backprops with random gradients.

**Recall:** If you want to compute the derivates, you can call the `.backward()` on a `Variable`. If `Variable` is a scalar *(i.e it holds a one element data)*, you don't need to specify any arguments to `backward()`, however if it has more elements, you need to specify a `grad_output` argument that is a tensor of matching shape.

In [7]:

```python
# Zero the gradient buffers
net.zero_grad()
# `grad_output` must have same shape as `out`
grad_output = torch.randn(1, 10)
# backprop on out
out.backward(grad_output)
```

**NOTE**

`torch.nn` only supports mini-batches. The entire `torch.nn` package only supports inputs that are a mini-batch of samples, and not a single sample.

For example, `nn.Conv2d` will take in a 4D Tensor of `nSamples x nChannels x Height x Width`.

If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.

**Recap:**

- `torch.Tensor` – A *multi-dimensional* array.
- `autograd.Variable` – Wraps a tensor and records the history of operations applied to it. Has the same API with `Tensor`, with some additions like `backward()`. Also holds the gradients w.r.t. the tensor.
- `nn.Module` – Neural Network module. *Convinent of encapsulating parameters*, with helpers for moving them to GPU, exporting, loading etc.
- `nn.Parameter` – A kind of `Variable` that is *automatically registered as a parameter when assigned an attribute* to a `Module`
- `autograd.Function` – Implements *forward and backward definitions of an autograd operation*. Every `Variable` operation creates at least a single `Function` node, that connects to the functions that created a `Variable` and encodes its history.

At this point, we covered:

- Defining a neural network
- Processing inputs and calling backward.

Still Left:

- Computing the loss
- Updating the weights of the network

### Loss Function

A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target.

There are several different [loss function](http://pytorch.org/docs/nn.html#loss-functions) under the `nn` package. A simple loss is `nn.MSELoss` which computes the mean-squared error between the input and the target.

In [8]:

```python
# A dummy example:
X_input = torch.randn(1, 32, 32)  # NOTE: No batch dimension ...use .unsqueeze!
X_input = Variable(X_input.unsqueeze(0))  # X_input.unsqueeze(0) adds an extra batch dim
target = Variable(torch.arange(1, 11))

# Output prediction by the network
output = net(X_input)

# Loss function criterion
loss_func = nn.MSELoss()

# calculate the loss
loss = loss_func(output, target)
print(loss)
```

```sh
Variable containing:
 37.4122
[torch.FloatTensor of size 1]


```

Now, if you follow the `loss` in the backward direction, using it's `.grad_fn` attribute, you'll see a computation graph that looks like this:

```python
input -> conv2d -> relu -> max_pool2d -> conv2d -> relu -> max_pool2d
      -> view -> linear -> relu -> linear -> relu -> linear -> softmax
      -> MSELoss
      -> loss

```

So, when we call the `loss.backward()`, the whole graph is differentiated w.r.t. the loss, and all Variables in the graphs have their `.grad` Variable accumulated with the gradient.

In [9]:

```python
# loss.grad_fn.next_functions
# ((<SoftmaxBackward object at 0x1095a48d0>, 0),)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Softmax
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # Linear
```

```sh
<MseLossBackward object at 0x10883c588>
<SoftmaxBackward object at 0x10883c438>
<AddmmBackward object at 0x10883c588>

```

### Backprop

To backpropagate the error, all we need to do is `loss.backward()`. You need to clear or zero out the existing gradients, else the gradients will be accumulated to existing gradients

Now, we call `loss.backward()` and we have a look at the 1st convolution bias before and after the call to backward.

In [10]:

```python
net.zero_grad()  # zero out the gradient buffer of all params

conv1_bias_before = net.conv1.bias.grad
print(f'Conv 1 bias before: {conv1_bias_before}')

loss.backward()

conv1_bias_after = net.conv1.bias.grad
print(f'Conv 1 bias afer: {conv1_bias_after}')
```

```sh
Conv 1 bias before: Variable containing:
 0
 0
 0
 0
 0
 0
[torch.FloatTensor of size 6]

Conv 1 bias afer: Variable containing:
1.00000e-03 *
  0.7892
  1.1216
 -4.2388
  1.7667
 -0.8663
 -0.6962
[torch.FloatTensor of size 6]


```

### Update the weights

The simplest update rul used in practise is the Stochastic Gradient Descent (SGD):

`weight = weight - learning_rate * gradient`

We can implement this using small Python code:

In [11]:

```python
learning_rate = 1e-2

for param in net.parameters():
    # .sub_ method mutates the param variable 
    # by subtracting its arguments from it.
    param.data.sub_(param.grad.data * learning_rate)
```

However, as you use neural networks, you want to use various different update rules such as *SGD*, *Nesterov-SGD*, *Adam*, *RMSProp*, etc. To enable this, the creators of Pytorch built a small package: `torch.optim` that implements all these methods. Using it is very simple:

In [12]:

```python
import torch.optim as optim
```

In [13]:

```python
# Create SGD optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-2)

# in your training loop:
optimizer.zero_grad()   # zeros the gradient buffer
output = net(X_input)
loss = loss_func(output, target)
loss.backward()
optimizer.step()  # Does the update
```

**Note**

Observe how gradient buffers had to be manually set to zero using `optimizer.zero_grad()`. This is because gradients are accumulated as explained in [Backprop section](http://localhost:8888/notebooks/pytorch-examples/03%20-%20Neural%20Networks/neural-networks.ipynb#Backprop).