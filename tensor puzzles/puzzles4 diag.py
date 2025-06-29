import torch
from utils import arange


def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]


def diag(a):
    # 使用索引的方法
    return a[arange(len(a)), arange(len(a))]


test = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
test_out = torch.zeros(3, dtype=torch.float32)
diag_spec(test, test_out)
print(test_out)
print(diag(test))