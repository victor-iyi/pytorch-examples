## Warm up – `NumPy`

Before introducing `PyTorch`, we will first implement the network using `numpy`.

Numpy provides an *n-dimensional* array object, and many functions for manipulating these arrays. Numpy is a generic framework for scientific computing; it does not know anything about computation graphs, or deep learning, or gradients. However we can easily use numpy to fit a two-layer network to random data by manually implementing the forward and backward passes through the network using numpy operations:

In [1]:

```python
# Typical NumPy import.
import numpy as np
```

In [2]:

```python
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
```

In [3]:

```python
# Create random input and output data.
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
```

In [4]:

```python
# Randomly initialize weights.
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
```

In [5]:

```python
# Learning rate.
lr = 1e-6
```

In [6]:

```python
# Training iterations.
train_iter = 500

for t in range(train_iter):
    # Foward pass.
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    
    # Compute loss: Squared error.
    loss = np.square(y - y_pred).sum()
    
    # Print loss and current time step.
    print(f'\rt = {t+1:,}\tloss = {loss:.4f}', end='')
    
    # Back propagation: Compute gradients
    # of w1 and w2 w.r.t. loss.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    # Update weights using Gradient Descent.
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
```

```sh
t = 500	loss = 0.000001655
```



## Warm up – `PyTorch`: Tensors

`Numpy` is a great framework, but it cannot utilize GPUs to accelerate its numerical computations. For modern deep neural networks, GPUs often provide speedups of [50x or greater](https://github.com/jcjohnson/cnn-benchmarks), so unfortunately `numpy` won’t be enough for modern deep learning.

Here we introduce the most fundamental `PyTorch` concept: *the Tensor*. **A PyTorch Tensor** is conceptually identical to a numpy array: a Tensor is an *n-dimensional* array, and PyTorch provides many functions for operating on these Tensors. Like numpy arrays, PyTorch Tensors do not know anything about deep learning or computational graphs or gradients; they are a generic tool for scientific computing.

However unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations. To run a PyTorch Tensor on GPU, you simply need to cast it to a new datatype.

Here we use PyTorch Tensors to fit a two-layer network to random data. Like the numpy example above we need to manually implement the forward and backward passes through the network:

In [1]:

```python
# Typical PyTorch import.
import torch
```

In [2]:

```python
# If you have cuda enabled with torch, use it
# otherwise, run on the CPU.
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
```

In [3]:

```python
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
```

In [4]:

```python
# Create random input and output data.
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)
```

In [5]:

```python
# Randomly initialize weights.
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)
```

In [6]:

```python
# Learning rate.
lr = 1e-6
```

In [7]:

```python
# Training iterations.
train_iter = 500

for t in range(train_iter):
    # Forward pass.
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    
    # Compute loss: Squared error.
    loss = (y_pred - y).pow(2).sum()
    print(f'\rt = {t+1:,}\tloss = {loss:.2f}', end='')
    
    # Back propagation: Compute gradients
    # of w1 and w2 w.r.t. loss.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    # Update weights using Gradient Descent.
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
```

```sh
t = 500	loss = 0.0083716

```



## Autograd

### PyTorch: Variables and autograd

In the above examples, we had to manually implement both the forward and backward passes of our neural network. Manually implementing the backward pass is not a big deal for a small two-layer network, but can quickly get very hairy for large complex networks.

Thankfully, we can use [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) to automate the computation of backward passes in neural networks. The `autograd` package in PyTorch provides exactly this functionality. When using `autograd`, the forward pass of your network will define a **computational graph**; nodes in the graph will be *Tensors*, and edges will be *functions* that produce output Tensors from input Tensors. Backpropagating through this graph then allows you to easily compute gradients.

This sounds complicated, it’s pretty simple to use in practice. We wrap our PyTorch Tensors in `Variable` objects; a `Variable` represents a node in a *computational graph*. If `x` is a `Variable` then `x.data` is a Tensor, and `x.grad` is another Variable holding the gradient of `x` with respect to some scalar value.

PyTorch Variables have the same API as PyTorch Tensors: (almost) any operation that you can perform on a Tensor also works on Variables; the difference is that using Variables defines a computational graph, allowing you to automatically compute gradients.

Here we use PyTorch Variables and `autograd` to implement our two-layer network; now we no longer need to manually implement the backward pass through the network:

In [1]:

```python
# Typical PyTorch import.
import torch

# Import Variable from the PyTorch's autograd package.
from torch.autograd import Variable
```

In [2]:

```python
# If you have cuda enabled with torch, use it
# otherwise, run on the CPU.
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
```

In [3]:

```python
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
```

In [4]:

```python
# Create random Tensors to hold inputs and outputs, and wrap them in
# Variables. Setting requires_grad to False indicates we don't need to
# compute gradients w.r.t. these Variables during the backward pass.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)
```

In [5]:

```python
# Create random Tensors for weights and wrap them in Variable. 
# Setting requires_grad to True indicates we want to compute
# gradients w.r.t. these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
```

In [6]:

```python
# Learning rate.
lr = 1e-6
```

In [7]:

```python
# Training iterations.
train_iter = 500

for t in range(train_iter):
    # Forward pass: compute predicted y using operations
    # on Variables; these are exactly the same operations we
    # used to compute the forward pass using Tensors, but
    # we don't need to keep references to intermediate values
    # since we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
    # Compute and print loss using operations on the Variables.
    # Now loss is a Varaible of shape (1,) and loss.data is a
    # Tensor of shape (1,); loss.data[0] is a scalar value holding
    # the loss.
    loss = (y_pred - y).pow(2).sum()
    print(f'\rt = {t+1:,}\tloss = {loss.data[0]:.2f}', end='')
    
    # Use the autograd to compute the backward pass. This will
    # compute the gradient of loss w.r.t. to all Variables with
    # requires_grad set to True. After this call, w1.grad and
    # w2.grad will be holding the gradients of the loss w.r.t.
    # w1 and w2 respectively.
    loss.backward()
    
    # Update weights using Gradient Descent; w1.data and w2.data are
    # Tensors, w1.grad and w2.grad are variables and w1.grad.data and
    # w2.grad.data are Tensors.
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    
    # Manually zero out the gradient buffer after updating 
    # the weights to prevent gradient accumulation.
    w1.grad.data.zero_()
    w2.grad.data.zero_()
```

```sh
t = 500	loss = 0.0074408

```



### PyTorch: Defining new autograd functions

Under the hood, each primitive *autograd* operator is really two functions that operate on Tensors. The `forward` function computes output Tensors from input Tensors. The `backward` function receives the gradient of the output Tensors w.r.t. some scalar value, and computes the gradient of the input Tensors w.r.t. that same scalar value.

In PyTorch, we can easily define our own *autograd operator* by defining a subclass of `torch.autograd.Function` and implementing the `forward` and `backward` functions. We can then use our new autograd operator by constructing an instance and calling it like a function, passing Variables containing input data.

In this example we define our own custom autograd function for performing the *ReLU nonlinearity*, and use it to implement our two-layer network:

In [1]:

```python
# Typical PyTorch import
import torch

# Import Variable from the PyTorch's autograd package.
from torch.autograd import Variable
```

In [2]:

```python
class MyReLU(torch.autograd.Function):
    """
    We can build our own custom autograd functions
    by creating a subclass of the `torch.autograd.Function`
    class and overriding the `forward` and `backward`
    static methods.
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we recieve a Tensor containing
        the input and return a tensor containing the computed
        output. In this case, we return the ReLU activation.
        
        `ctx` is a context object that can be used to stack 
        information for backward computation. You can cache 
        arbitrary objects for use in the backward pass using 
        `ctx.save_for_backward` method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        We recieve the gradient of the loss w.r.t. the output.
        Now, we compute the gradient of the loss w.r.t. the input.
        
        `ctx` is a context object that can also be used to get
        stored information in the forward pass. The saved Tensors
        is stored in the `ctx.saved_tesnors`. The `ctx.saved_tensors`
        returns a tuple of saved tensors.
        
        Since we saved a single Tensor, the saved_tesnor contains
        a single value therefore we unpack it by having a comma (,)
        after the variable name:
        
        >>> names = ('John',)  # Comma after the 1st element.
        >>> john, = names
        >>> print(john)
        'John'
        
        >>> names = ('John')  # No comma
        >>> (john) = names
        >>> print(john)
        'John'
        
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0  # ReLU derivative.
        return grad_input
```

In [3]:

```python
# If you have cuda enabled with torch, use it
# otherwise, run on the CPU.
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
```

In [4]:

```python
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
```

In [5]:

```python
# Create random Tensors to hold inputs and outputs, and wrap them in
# Variables. Setting requires_grad to False indicates we don't need to
# compute gradients w.r.t. these Variables during the backward pass.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)
```

In [6]:

```python
# Create random Tensors for weights and wrap them in Variable. 
# Setting requires_grad to True indicates we want to compute
# gradients w.r.t. these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
```

In [7]:

```python
# Learning rate.
lr = 1e-6
```

In [8]:

```python
# Training iterations.
train_iter = 500

for t in range(train_iter):
    # We don't create an instance of our custom class,
    # instead we use the `Function.apply` to apply our 
    # custom activation function.
    # We apply because we want PyTorch to still keep
    # the computations inside the current graph, and
    # it's history.
    relu = MyReLU.apply
    
    # Forward pass: We multiply our input by the 1st weight
    # matrix (w1) then we use our custom non-linearity
    # then matrix multiply the 2nd weight to get our prediction.
    y_pred = relu(x.mm(w1)).mm(w2)
    
    # Compute and print loss using operations on the Variables.
    loss = (y_pred - y).pow(2).sum()
    print(f'\rt = {t+1:,}\tloss = {loss.data[0]:.2f}', end='')
    
    # Use the autograd to compute the backward pass. 
    loss.backward()
    
    # Update the weights using Gradient Descent.
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    
    # Manually zero out the gradient buffer after updating 
    # the weights to prevent gradient accumulation.
    w1.grad.data.zero_()
    w2.grad.data.zero_()
```

```sh
t = 500	loss = 0.0097190

```



### TensorFlow: Static Graphs

PyTorch autograd looks a lot like TensorFlow: in both frameworks we define a computational graph, and use automatic differentiation to compute gradients. The biggest difference between the two is that TensorFlow’s computational graphs are **static** and PyTorch uses **dynamic computational graphs**.

In TensorFlow, we define the computational graph once and then execute the same graph over and over again, possibly feeding different input data to the graph. In PyTorch, each forward pass defines a new computational graph.

Static graphs are nice because you can optimize the graph up front; for example a framework might decide to fuse some graph operations for efficiency, or to come up with a strategy for distributing the graph across many GPUs or many machines. If you are reusing the same graph over and over, then this potentially costly up-front optimization can be amortized as the same graph is rerun over and over.

One aspect where static and dynamic graphs differ is control flow. For some models we may wish to perform different computation for each data point; for example a recurrent network might be unrolled for different numbers of time steps for each data point; this unrolling can be implemented as a loop. With a static graph the loop construct needs to be a part of the graph; for this reason TensorFlow provides operators such as `tf.scan` for embedding loops into the graph. With dynamic graphs the situation is simpler: since we build graphs on-the-fly for each example, we can use normal **imperative flow control** to perform computation that differs for each input.

To contrast with the PyTorch autograd example above, here we use TensorFlow to fit a simple two-layer net:

In [1]:

```python
# Typical NumPy import.
import numpy as np

# Standard way to import TensorFlow.
import tensorflow as tf
```

In [2]:

```python
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 100, 1000, 10
```

In [3]:

```python
# Create placeholders for the input and output,
# to serve as gateway for feeding inputs and outputs
# to the network during execution.
x = tf.placeholder(tf.float32, shape=[N, D_in])
y = tf.placeholder(tf.float32, shape=[N, D_out])
```

In [4]:

```python
# Randomly initialize learnable weights. A TensorFlow
# Variable persists it's value across execution of the graph.
w1 = tf.Variable(tf.random_normal(shape=[D_in, H]))
w2 = tf.Variable(tf.random_normal(shape=[H, D_out]))
```

In [5]:

```python
# Learning rate.
lr = 1e-6
```

In [6]:

```python
# Forward pass: Propagate the input through the network
# by performing some operations on TensorFlow's Tensors.
# NOTE: No operation is actually being run at this point,
# we're just setting up the computational graph that'll
# be executed/run later on.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)
```

In [7]:

```python
# Compute the loss: Three different ways to compute Squared error
# in TensorFlow.
loss = tf.reduce_sum(tf.squared_difference(y_pred, y))
# loss = tf.reduce_sum(tf.square(y - y_pred))
# loss = tf.reduce_sum((y - y_pred) ** 2.0)

print(loss)  # Prints the node that holds the operation on loss.
```

```sh
Tensor("Sum:0", shape=(), dtype=float32)

```

In [8]:

```python
# Compute the gradient of the loss, w.r.t. w1 & w2
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
```

In [9]:

```python
# Update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# in TensorFlow the the act of updating the value of the weights is part of
# the computational graph; in PyTorch this happens outside the computational
# graph.
new_w1 = w1.assign(w1 - lr * grad_w1)
new_w2 = w2.assign(w2 - lr * grad_w2)
```

In [10]:

```python
# It's time to run our computational graph.
# We run graphs using the TensorFlow's Session.
with tf.Session() as sess:
    # Run the graph ones to initialize the Variables w1 & w2.
    sess.run(tf.global_variables_initializer())
    
    # Create a NumPy array that holds our actual data
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    
    # Training iterations.
    train_iter = 500
    
    for t in range(train_iter):
        # Execute the graph many times. Each time it executes we want to bind
        # x_value to x and y_value to y, specified with the feed_dict argument.
        # Each time we execute the graph we want to compute the values for loss,
        # new_w1, and new_w2; the values of these Tensors are returned as numpy
        # arrays.
        _loss, _, _ = sess.run([loss, new_w1, new_w2], 
                               feed_dict={ x: x_value, y: y_value })
        # Print training progress.
        print(f'\rt = {t+1:,}\tLoss = {_loss:.2f}', end='')
```

```sh
t = 500	Loss = 0.00646100

```



## `nn.Module`

### PyTorch: nn

Computational graphs and autograd are a very powerful paradigm for defining complex operators and automatically taking derivatives; however for large neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the computation into **layers**, some of which have *learnable parameters* which will be optimized during learning.

In TensorFlow, packages like `Keras`, `TensorFlow-Slim`, and `TFLearn` provide higher-level abstractions over raw computational graphs that are useful for building neural networks.

In PyTorch, the `nn` package serves this same purpose. The `nn` package defines a set of **Modules**, which are roughly equivalent to neural network layers. A Module receives input Variables and computes output Variables, but may also hold internal state such as Variables containing learnable parameters. The `nn` package also defines a set of useful loss functions that are commonly used when training neural networks.

In this example we use the `nn` package to implement our two-layer network:

