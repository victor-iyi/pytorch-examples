## What is PyTorch?

It's a Python based scientific computing package targeted at two sets of audiences:

- A replacement for `numpy` to use the power of GPUs
- A deep learning research platform that provides maximum flexibility and speed

## Getting Started

### Tensors

In [1]:

```python
import torch
```

In [2]:

```python
# Construct a 5x3 matrix
x = torch.Tensor(5, 3)
print(x)
```

```sh
1.00000e-29 *
  0.0000  0.0000  3.9260
  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000
[torch.FloatTensor of size 5x3]


```

In [3]:

```python
# Construct a 5x3 randomly initialized matrix
x = torch.rand(5, 3)
print(x)
```

```sh
 0.9152  0.4046  0.7690
 0.6653  0.6564  0.9553
 0.4627  0.0858  0.8089
 0.2361  0.9341  0.0296
 0.0496  0.9051  0.3340
[torch.FloatTensor of size 5x3]


```

In [4]:

```python
# get it's size
x.size()
```

```sh
torch.Size([5, 3])
```

In [5]:

```python
y = torch.FloatTensor(3, 5)
print(y)
```

```sh
 0.0000e+00  0.0000e+00  2.6258e-29  3.6893e+19  5.6052e-45
 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00
 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00  2.6372e-29
[torch.FloatTensor of size 3x5]


```



### Operations with `pytorch`

### Addition

In [6]:

```python
x = torch.Tensor(5, 3)
y = torch.rand(5, 3)
```

In [7]:

```python
print(x + y)
```

```sh
 0.4040  0.5859  0.8088
 0.2688  0.1638  0.1029
 0.0107  0.7332  0.3672
 0.2479  0.1942  0.1848
 0.3706  0.5134  0.4678
[torch.FloatTensor of size 5x3]


```

In [8]:

```python
out = x + y
print(out)
```

```sh
 0.4040  0.5859  0.8088
 0.2688  0.1638  0.1029
 0.0107  0.7332  0.3672
 0.2479  0.1942  0.1848
 0.3706  0.5134  0.4678
[torch.FloatTensor of size 5x3]


```

In [9]:

```python
# syntax 2:
print(torch.add(x, y))
```

```sh
 0.4040  0.5859  0.8088
 0.2688  0.1638  0.1029
 0.0107  0.7332  0.3672
 0.2479  0.1942  0.1848
 0.3706  0.5134  0.4678
[torch.FloatTensor of size 5x3]


```

In [10]:

```python
out = torch.add(x, y)
print(out)
```

```sh
 0.4040  0.5859  0.8088
 0.2688  0.1638  0.1029
 0.0107  0.7332  0.3672
 0.2479  0.1942  0.1848
 0.3706  0.5134  0.4678
[torch.FloatTensor of size 5x3]


```

In [11]:

```python
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)
```

```sh
 0.4040  0.5859  0.8088
 0.2688  0.1638  0.1029
 0.0107  0.7332  0.3672
 0.2479  0.1942  0.1848
 0.3706  0.5134  0.4678
[torch.FloatTensor of size 5x3]


```

In [12]:

```python
# in-place
y.add_(x)
print(y)
```

```sh
 0.4040  0.5859  0.8088
 0.2688  0.1638  0.1029
 0.0107  0.7332  0.3672
 0.2479  0.1942  0.1848
 0.3706  0.5134  0.4678
[torch.FloatTensor of size 5x3]


```

**NOTE:** Any operation that mutates a tensor in-pace is post-fixed with an `_`. For example: `x.copy_(y)`, `x.t_()`, will change `x`.

In [13]:

```python
# You can use standard NumPy-like indexing
print(x[:, 1])
```

```sh
1.00000e-45 *
  0.0000
  7.0065
  0.0000
  0.0000
  0.0000
[torch.FloatTensor of size 5]


```

### Reshaping: `Tensor.view(*shape)`

In [14]:

```python
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions

print('x.size() = {}'.format(x.size()))
print('y.size() = {}'.format(y.size()))
print('z.size() = {}'.format(z.size()))
```

```sh
x.size() = torch.Size([4, 4])
y.size() = torch.Size([16])
z.size() = torch.Size([2, 8])

```

In [15]:

```python
# Tensor.view_as(Tensor)
a = x.view_as(z)  # inferred from a Tensor shape
print('a.size() = {}'.format(a.size()))
```

```sh
a.size() = torch.Size([2, 8])

```



In [16]:

```python
# Taking a dot product of x with y using the Python 3.5 syntax
product = x@y
print(product)
```

```sh
 0.0000e+00  0.0000e+00  2.4032e-29  3.3767e+19  2.0280e-29
 0.0000e+00  0.0000e+00  1.7468e-29  2.4544e+19  2.5193e-29
 0.0000e+00  0.0000e+00  1.2150e-29  1.7072e+19  2.1332e-29
 0.0000e+00  0.0000e+00  6.1984e-30  8.7092e+18  7.8053e-31
 0.0000e+00  0.0000e+00  1.3022e-30  1.8297e+18  8.8081e-30
[torch.FloatTensor of size 5x5]


```

## NumPy Bridge

Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

**NOTE:** The Torch Tensor and NumPy array will share their underlying memory locations, and changing one will change the other.

In [17]:

```python
import numpy as np
```

In [18]:

```python
a = torch.ones(5)
print(a)
print(b)
```

```bash
 1
 1
 1
 1
 1
[torch.FloatTensor of size 5]


```

In [19]:

```python
b = a.numpy()
print(b)
```

```bash
[1. 1. 1. 1. 1.]

```

See how the numpy array changed in value

In [20]:

```python
a.add_(1)
print(a)
print(b)
```

```bash
 2
 2
 2
 2
 2
[torch.FloatTensor of size 5]

[2. 2. 2. 2. 2.]

```

In [21]:

```python
# b is a numpy array from the torch Tensor, a
# c is a torch tensor created from the numpy array, b
# a, b & c shares the same memory location
c = torch.from_numpy(b)
print(c)
```

```bash
 2
 2
 2
 2
 2
[torch.FloatTensor of size 5]


```

In [22]:

```python
a.add_(3)
print(f'a: {a} -Torch')
print(f'b: {b} -Numpy')
print(f'c: {c} -Torch')
```

```bash
a: 
 5
 5
 5
 5
 5
[torch.FloatTensor of size 5]
 -Torch
    
b: [5. 5. 5. 5. 5.] -Numpy
    
c: 
 5
 5
 5
 5
 5
[torch.FloatTensor of size 5]
 -Torch

```

### Converting NumPy array to Torch Tensor

See how the change in the `np array` changed the Torch Tensor automatically

In [23]:

```python
a = np.ones(5)
b = torch.from_numpy(a)
# Add 1 to a and store the result in a
# i.e. Update the value of a by adding 1 to it.
np.add(a, 1, out=a)
print(a)
```

```bash
[2. 2. 2. 2. 2.]

 2
 2
 2
 2
 2
[torch.DoubleTensor of size 5]


```

## CUDA Tensors

Tensors can be moved into the GPU using the `.cuda` method.

In [24]:

```python
# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    # Create `x` as a Float tensor on the GPU
    x = torch.cuda.FloatTensor(5, 3)
    # Randomly initialize y ont the GPU
    y = torch.rand(5, 3).cuda()
    result = torch.Tensor(5, 3).cuda()
    # Add a & b on the GPU
    torch.cuda.torch.add(a, b, out=result)
    # PRINT THE RESULT
    print(x + y)
else:
    print("You don't have a GPU or CUDA  isn't enabled!")
```

```bash
You don't have a GPU or CUDA  isn't enabled!

```
