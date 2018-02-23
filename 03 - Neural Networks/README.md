## Neural Networks

Neural networks can be constructed using the `torch.nn` package.

Now that you had a glipse of `autograd`, `nn` depends on `autograd` to define models and differentiate them. An `nn.Module` contains layers, and a method `forward(input)` that returns the `output`.

For example, look at a simple convolutional neural network that classifies digit images:

![Convnet](../images/mnist.png)

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
        # Max pooling over a 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten
        x = x.view(-1, self.num_flat_features(x))
        # Apply relu to the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
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
print(params[0].size())  # 1st layer (conv1) parameters
```

```sh
torch.Size([6, 1, 5, 5])

```

In [5]:

```python
# Weights and bias are stored in `net.Module.weight`, `net.Module.bias`
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
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)
```

```sh
Variable containing:
 0.1151 -0.0960  0.0421 -0.0158 -0.0466  0.0141  0.0903 -0.0087 -0.0068 -0.0138
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

### NOTE

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