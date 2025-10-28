import torch
from torch import nn

# 定义输入数据，尺寸为[1, 3, 2, 2]，包括了3个通道，每个通道的图像大小是2*2的
input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                       [[1.0, 1.0], [0.0, 0.0]],
                       [[2.0, 2.0], [2.0, 2.0]]]])

# 创建实例归一化层，因为输入的数据有3个通道，所以num_features=3
ins = nn.InstanceNorm2d(num_features=3, eps=1e-5)

output = ins(input) # 应用实例归一化
print ("InstanceNorm2d: ")
print (output) # 打印结果







