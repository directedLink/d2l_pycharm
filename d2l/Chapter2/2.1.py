import torch

x = torch.arange(15)
print(x)

print(x.shape)
print(x.numel())

X = x.reshape(3, 5)
print(X)

print(torch.zeros((3, 4, 5)))

print(torch.ones((3, 4, 5)))

print(torch.rand(2, 4, 5))

print(torch.tensor([[2, 3], [3, 4]]))


x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

print(x + y, x - y, x * y, x / y, x ** y)

print(torch.exp(x))


print("                                              ")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 2, 3, 1]])

print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

print(X > Y)

print(X.sum(),
      Y.sum())

a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a, b)
b = b + 1
print(b)
print(a + b)

print(X[-1], X[1:3])

X[2, 3] = 666
print(X)

X[0:1, :] = 888
print(X)

before = id(Y)
print(before)
Y = Y + X
print(id(Y))
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = Y + X
print('id(Z):', id(Z))

before = id(Y)
print(id(Y))
Y += X
print(id(Y))

A = X.numpy()
B = torch.tensor(A)

print(type(A), type(B))


a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))