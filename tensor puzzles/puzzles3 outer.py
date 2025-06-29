from utils import arange
import torch

def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]

def outer(a, b):
    # 使用广播的方法进行计算
    return a[..., None] * b[None, :]
            

test_a = torch.tensor([1, 2, 3], dtype=torch.float32)
test_b = torch.tensor([4, 5, 6], dtype=torch.float32)
test_out = torch.zeros((3, 3), dtype=torch.float32)
outer_spec(test_a, test_b, test_out)
print(test_out)
print(outer(test_a, test_b))