import torch
from torch import nn
# 定义3*3的输入数据
input = torch.tensor(data = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
                     dtype = torch.float32)

up1 = nn.Upsample(scale_factor = 1, mode='nearest')
output = up1(input) # 应用上采样
print ("Upsample(scale_factor = 1): ")
print (output)
up2 = nn.Upsample(scale_factor = 2, mode='nearest')
output = up2(input) # 应用上采样
print ("Upsample(scale_factor = 2): ")
print (output)
up3 = nn.Upsample(scale_factor = 3, mode='nearest')
output = up3(input) # 应用上采样
print ("Upsample(scale_factor = 3): ")
print (output)




