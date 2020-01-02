
from __future__ import print_function
import torch
import numpy as np


x = torch.empty(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
z = torch.zeros(5, 3, dtype=torch.long)
print(z)
p = torch.tensor([5.5, 3])
print(p)


x = x.new_ones(5, 4, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
print(x.size())




y = torch.rand(5, 4)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 4)
torch.add(x, y, out=result)
print(result)
y.add_(x)
print(y)



a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)



if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
