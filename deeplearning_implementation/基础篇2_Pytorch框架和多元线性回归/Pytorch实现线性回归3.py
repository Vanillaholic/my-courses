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

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

if __name__ == '__main__':
    # 使用read_excel，读取训练数据train.xlsx
    df = pd.read_excel("train.xlsx")
    # 获取训练数据的特征
    feature_names = [col for col in df.columns if col not in ['price']]
    # 打印特征的数量
    print(f"Feature num: {len(feature_names)}")

    # 将输入特征，转换为张量x
    x = torch.Tensor(df[feature_names].values)
    # 训练标签，转换为张量y
    y = torch.Tensor(df['price'].values).unsqueeze(1)
    # 打印训练数据的数量
    print(f"Training samples: {len(x)}")

    # 在使用Pytorch训练模型时，需要创建三个对象
    # 第1个是模型本身model，它就是我们设计的线性回归模型
    model = LinearRegression()
    # 第2个是优化器optimizer，它用来优化模型中的参数
    # 最常使用的就是Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 第3个是损失函数criterion，对于回归问题，使用MSELoss，均方误差
    criterion = nn.MSELoss()

    # 进入模型的迭代循环，使用的是批量梯度下降算法
    # 每次迭代，都会基于全部样本计算损失值，执行梯度下降
    # 这里循环的轮数定为2万，可以使模型达到收敛
    for epoch in range(20000):
        # 在循环中，包括了5个步骤
        h = model(x) # 1.计算模型的预测值h
        loss = criterion(h, y) # 2.计算预测h和标签y之间的损失loss
        loss.backward() # 3.使用backward计算梯度
        optimizer.step() # 4.使用optimizer.step更新参数
        optimizer.zero_grad() # 5.将梯度清零
        # 这5个步骤，是使用pytorch框架训练模型的定式

        # 每迭代1000次，就打印一次模型的损失，用于观察训练的过程
        if (epoch + 1) % 1000 == 0:
            # 打印迭代轮数和损失外
            print(f'After {epoch + 1} iterations, Train Loss: {loss.item():.3f}')

            h_np = h.detach().numpy()
            y_np = y.detach().numpy()
            # 打印MSE、MAE和R2这三个回归指标
            mse = mean_squared_error(y_np, h_np)
            mae = mean_absolute_error(y_np, h_np)
            r2 = r2_score(y_np, h_np)
            print(f'\tMean Squared Error: {mse:.3f}')
            print(f'\tMean Absolute Error: {mae:.3f}')
            print(f'\tR2 Score: {r2:.3f}')

    # 打印模型训练出的参数
    print(model.state_dict())
    # 将模型参数保存到文件lr.pth中
    # lr.pth就是我们最终训练得到的模型
    torch.save(model.state_dict(), 'lr.pth')

