import torch
from torch import nn
# 定义3*3的输入数据
input = torch.tensor(data = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
                     dtype = torch.float32)

pad1 = nn.ReflectionPad2d(padding = 1) # 定义padding=1的反射填充
output = pad1(input) # 执行反射填充
print ("ReflectionPad2d(padding = 1): ")
print (output) # 打印结果
print("")
pad2 = nn.ReflectionPad2d(padding = 2) # 定义padding=2的反射填充
output = pad2(input) # 执行反射填充
print ("ReflectionPad2d(padding = 2): ")
print (output) # 打印结果







