
import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z)
print(out)


# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a)
# b = (a * a).sum()
# print(a.requires_grad)
# print(b.grad_fn)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)


out.backward()
# d(out)/dx
print(x.grad)
