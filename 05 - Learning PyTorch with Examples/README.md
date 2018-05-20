### Learning PyTorch with Examples

#### Author: [Justin Johnson](https://github.com/jcjohnson/)

This tutorial introduces the fundamental concepts of [PyTorch](https://github.com/pytorch/pytorch) through self-contained examples.

At its core, PyTorch provides two main features:

- An n-dimensional Tensor, similar to numpy but can run on GPUs
- Automatic differentiation for building and training neural networks

We will use a fully-connected ReLU network as our running example. The network will have a single hidden layer, and will be trained with gradient descent to fit random data by minimizing the Euclidean distance between the network output and the true output.

**Note:** You can browse the individual examples at the end of this page.

### Table of Contents

- [Tensors](/01%20-%20Tensors.ipynb)

  - [Warm-up: numpy](/01%20-%20Tensors.ipynb#Warm-up-%E2%80%93-NumPy)
  - [PyTorch: Tensors](/01%20-%20Tensors.ipynb#Warm-up-%E2%80%93-PyTorch:-Tensors)

  ​

- [Autograd](/02%20-%20Autograd.ipynb)

  - [PyTorch: Variables and autograd](/02%20-%20Autograd.ipynb#PyTorch:-Variables-and-autograd)
  - [PyTorch: Defining new autograd functions](/02%20-%20Autograd.ipynb#PyTorch:-Defining-new-autograd-functions)
  - [TensorFlow: Static Graphs](/02%20-%20Autograd.ipynb#TensorFlow:-Static-Graphs)

  ​

- [nn module](/03%20-%20nn.Module.ipynb)

  - [PyTorch: nn](/03%20-%20nn.Module.ipynb#PyTorch:-nn)
  - [PyTorch: optim](/03%20-%20nn.Module.ipynb#PyTorch:-optim)
  - [PyTorch: Custom nn Modules](/03%20-%20nn.Module.ipynb#PyTorch:-Custom-nn-Modules)
  - [PyTorch: Control Flow + Weight Sharing](/03%20-%20nn.Module.ipynb#PyTorch:-Control-Flow-+-Weight-Sharing)