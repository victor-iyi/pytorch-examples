### Learning PyTorch with Examples

#### Author: [Justin Johnson](https://github.com/jcjohnson/)

This tutorial introduces the fundamental concepts of [PyTorch](https://github.com/pytorch/pytorch) through self-contained examples.

At its core, PyTorch provides two main features:

- An n-dimensional Tensor, similar to numpy but can run on GPUs
- Automatic differentiation for building and training neural networks

We will use a fully-connected ReLU network as our running example. The network will have a single hidden layer, and will be trained with gradient descent to fit random data by minimizing the Euclidean distance between the network output and the true output.

**Note:** You can browse the individual examples at the end of this page.

### Table of Contents

- [Tensors](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/01%20-%20Tensors.ipynb)

  - [Warm-up: numpy](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/01%20-%20Tensors.ipynb#Warm-up-%E2%80%93-NumPy)
  - [PyTorch: Tensors](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/01%20-%20Tensors.ipynb#Warm-up-%E2%80%93-PyTorch:-Tensors)

  ​

- [Autograd](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/02%20-%20Autograd.ipynb)

  - [PyTorch: Variables and autograd](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/02%20-%20Autograd.ipynb#PyTorch:-Variables-and-autograd)
  - [PyTorch: Defining new autograd functions](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/02%20-%20Autograd.ipynb#PyTorch:-Defining-new-autograd-functions)
  - [TensorFlow: Static Graphs](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/02%20-%20Autograd.ipynb#TensorFlow:-Static-Graphs)

  ​

- [nn module](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/03%20-%20nn.Module.ipynb)

  - [PyTorch: nn](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/03%20-%20nn.Module.ipynb#PyTorch:-nn)
  - [PyTorch: optim](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/03%20-%20nn.Module.ipynb#PyTorch:-optim)
  - [PyTorch: Custom nn Modules](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/03%20-%20nn.Module.ipynb#PyTorch:-Custom-nn-Modules)
  - [PyTorch: Control Flow + Weight Sharing](http://localhost:8888/notebooks/pytorch-examples/05%20-%20Learning%20PyTorch%20with%20Examples/03%20-%20nn.Module.ipynb#PyTorch:-Control-Flow-+-Weight-Sharing)