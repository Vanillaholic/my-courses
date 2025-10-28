import torch
from torch import nn

# nn.BatchNorm2d的输入，是一个4维的张量
# 这个输入与卷积层conv2d的输入是一致的
# 输入张量的4个维度:
# 分别代表批量数batch_size
# 通道数channels
# 特征图的高height和宽width
x = torch.randn(10, 3, 5, 5)
print(f"x shape = {x.shape}")

# 指定bn层的参数，输入特征数num_features=3
# 每个通道都会独立的进行批量归一化计算
# 3个通道，就会计算3次批量归一化
bn = nn.BatchNorm2d(num_features = 3)
# 打印出bn层的参数，由于特征数为3
# 因此对应了3个缩放w和3个偏置b参数
print("BatchNorm weight (gamma):")
print(bn.weight)
print("BatchNorm bias (beta):")
print(bn.bias)
# 将x传入bn层，可以计算出结果y
y = bn(x)
print(f"y shape = {y.shape}")










