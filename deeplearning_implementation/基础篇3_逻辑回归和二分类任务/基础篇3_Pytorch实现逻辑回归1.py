import torch

# 基于模型结构，实现逻辑回归模型
# 首先定义类LogisticRegression，继承nn.Module类
class LogisticRegression(torch.nn.Module):
    # init函数用于初始化模型
    # 函数传入参数n，代表输入特征的数量
    def __init__(self, n):
        super(LogisticRegression, self).__init__()
        # 定义一个线性层，该线性层输入n个特征，输出1个结果
        self.layer = torch.nn.Linear(n, 1)

    # forward函数用于定义模型前向传播的计算逻辑
    # 函数传入数据x
    def forward(self, x):
        z = self.layer(x) # 将x输入至线性层
        # 将结果z输入至sigmoid函数，计算出逻辑回归的输出
        h = torch.sigmoid(z)
        return h #返回结果h

# 在main函数中实现模型的测试代码
if __name__ == '__main__':
    # 创建模型model，输入特征的个数为3
    model = LogisticRegression(3)
    print(model) # 打印model，可以看到模型的结构
    print("")

    # 接着使用循环，遍历模型中的参数
    for name, param in model.named_parameters():
        # 打印参数名name和参数的尺寸param.data.shape
        print(f"{name}: {param.data.shape}")
    print("")

    # 定义一个100×3大小的张量
    # 代表了100个输入数据，每个数据包括3个特征值
    x = torch.zeros([100, 3])
    h = model(x) # 将x输入至模型model，得到预测结果h

    print(f"x: {x.shape}")
    print(f"h: {h.shape}")


