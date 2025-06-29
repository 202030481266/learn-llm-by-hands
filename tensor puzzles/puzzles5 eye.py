import torch
from utils import arange



def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1
        
def eye(n):
    # 依然使用广播的方法进行计算
    # 但是这里有一个很巧妙的思维转换
    return (arange(n)[None, :] == arange(n)[:, None]) * 1
    

test = torch.zeros((3, 3), dtype=torch.float32)
test_out = torch.zeros((3, 3), dtype=torch.float32)
eye_spec(test_out)
print(test_out)
print(eye(3))