from utils import arange
import torch


def sum_spec(a, out):
    out[0] = 0
    for i in range(len(a)):
        out[0] += a[i]


def sum(a):
    # 使用内积的方法计算
    ones = (a * 0 + 1)[..., None] # [n, 1]
    return (a[None, :] @ ones)[0]



test = torch.tensor([1, 3, 5], dtype=torch.float32)
test_out = torch.zeros(1)
sum_spec(test, test_out)
print(test_out)
print(sum(test))