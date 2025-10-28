import torch
from torch import nn

# 定义模型类 LinearRegression，它继承了PyTorch的nn.Module类
# nn.Module是所有模型的基类，包括了模型的基本功能
class LinearRegression(nn.Module):
    # init函数用于初始化模型的结构和参数
    def __init__(self):
        super().__init__()
        # 对于房价预测这个问题，有12个输入特征和1个输出结果
        self.layer = nn.Linear(12, 1)

    # forward函数用于定义模型前向传播的计算逻辑
    def forward(self, x):
        # 输入的特征向量是x，将x传入至layer进行计算
        # 这个过程相当于计算线性回归的方程h(x)
        return self.layer(x)

if __name__ == '__main__':
    model = LinearRegression() # 创建模型
    print(model) # 打印model，可以看到模型的结构
    print("")

    # 使用循环，遍历模型中的参数
    for name, param in model.named_parameters():
        # 打印参数名name和参数的值param.data
        print(f"{name}: {param.data}")
    print("")

    # 定义一个100×12大小的张量
    # 代表了100个数据，每个数据由12个特征值
    x = torch.zeros([100, 12])
    h = model(x) # 将x输入至模型model，得到预测结果h

    print(f"x: {x.shape}")
    print(f"h: {h.shape}")
















