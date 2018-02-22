# Autograd: automatic differentiation

The `autograd` package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that very single iteration can be different.

## Variable

`autograd.Variable` is the central class of the package. It wraps a Tensor, and supports nearly all of the operations defined on it. Once you finish your computation you can call `.backward()` and have all the gradients computed automatically.

You can access the raw tensor through the `.data` attribute, whie the gradient w.r.t. this variable is accumulated into `.grad`

`autograd.Variable`

- data
- grad
- creator

`Variable` and `Function` are interconnected and build up an acyclic graph, that encodes a complete history of computation. Each variable has a `.grad_fn` attribute that references a `Function` that has created the `Variable` (except for Variables created by the user –their `grad_fn` is `None`).

If you want to compute the derivates, you can call the `.backward()` on a `Variable`. If `Variable` is a scalar *(i.e it holds a one element data),* you don't need to specify any arguments to `backward()`, however if it has more elements, you need to specify a `grad_output` argument that is a tensor of matching shape.



In [1]:

```python
import torch
from torch.autograd import Variable
```

In [2]:

```python
# Create a Variable:
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)
```

```bash
Variable containing:
 1  1
 1  1
[torch.FloatTensor of size 2x2]


```

`x` is a user created Varaible, therefore, it's `grad_fn` is `None`

In [3]:

```python
print(x.grad_fn)
```

```bash
None

```

In [4]:

```python
# Do an operation of variable:
y = x + 2
print(y)
```

```bash
Variable containing:
 3  3
 3  3
[torch.FloatTensor of size 2x2]


```

`y` was created as a result of an operation, so it has a `grad_fn`.

In [5]:

```python
print(y.grad_fn)
```

```bash
<AddBackward0 object at 0x108619b38>

```

In [6]:

```python
# Do more operation on y
z = y * y * 3

out = z.mean()

print(z)
print(out)
```

```bash
Variable containing:
 27  27
 27  27
[torch.FloatTensor of size 2x2]

Variable containing:
 27
[torch.FloatTensor of size 1]


```

## Gradients

Let's backprop now `out.backward()` is equivalent to doing `out.backward(torch.Tensor([1.0]))`

In [7]:

```python
out.backward()
```

In [8]:

```python
# print gradients d(out)/dx
print(x.grad)
```

```bash

Variable containing:
 4.5000  4.5000
 4.5000  4.5000
[torch.FloatTensor of size 2x2]

```

You should have got a matrix of `4.5`. Let's call the `out` Variable $“o”$.
$$
We \space have \space that; \space o = {1\over4} \sum_i z_i, z_i = 3(x_i + 2)^2 \space and \space z_i|_{x=1} = 27.
$$

$$
Therefore, {\delta o \over \delta x_i} = {3\over2}(x_i + 2) ,\space hence \space {\delta o \over \delta x_i}|_{x_i=1}~ =  {9\over2} = 4.5
$$

You can do many crazy things with `autograd`!

In [9]:

```python
# Randomly initialze x
x = torch.randn(3)
x = Variable(x, requires_grad=True)
```

In [10]:

```python
# y is the element-wise multiplication of x & 2
y = x * 2
while y.data.norm() < 1000:
    y *= 2
print(y)
```

```bash
Variable containing:
 1169.9318
  291.7206
 -428.9764
[torch.FloatTensor of size 3]


```

In [11]:

```python
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)
```

```bash
Variable containing:
  409.6000
 4096.0000
    0.4096
[torch.FloatTensor of size 3]


```

### Read Later:

Documentaion of `Variable` and `Function` is at <http://pytorch.org/docs/autograd>