import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x - y, x / y, x ** y)

x = torch.arange(6)
print(x,
      x[3],
      len(x))

print(x.shape)

A = torch.arange(20).reshape(5, 4)
print(A)

print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)

X = torch.arange(24).reshape(2, 3, 4)
print(X)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A, A + B)
print(A * B)

a = 2
X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(a + X, a * X, (a + X).shape)

x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())

print(A, A.shape, A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)

print(A.sum(axis=[0, 1]))

print(A.mean(), A.sum() / A.numel())

print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

sum_A = A.sum(axis=1, keepdim=True)
print(sum_A)
print(A / sum_A)
print(A.cumsum(axis=1))

y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x * y))

print(A.shape, x.shape, torch.mv(A, x))

B = torch.ones(4, 3)
print(torch.mm(A, B))

u = torch.tensor([3.0, -4.0])
print(u.norm(),
      torch.abs(u).sum())

print(torch.norm(torch.ones((4, 9))))