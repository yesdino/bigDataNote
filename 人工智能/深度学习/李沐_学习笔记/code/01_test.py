import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# print("X:\n{}".format(X))

a = X[::2, ::3]
print("a:\n{}".format(a))
# print("b:\n{}".format(b))
