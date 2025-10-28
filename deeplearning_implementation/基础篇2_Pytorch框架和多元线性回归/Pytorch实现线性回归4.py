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
    # 测试的流程与训练差不多
    # 首先需要读取测试数据集test.xlsx
    df = pd.read_excel("test.xlsx")
    feature_names = [col for col in df.columns if col not in ['price']]
    x = torch.Tensor(df[feature_names].values)
    y = torch.Tensor(df['price'].values).unsqueeze(1)
    # 包括了690个样本
    print(f"Testing samples: {len(x)}")

    model = LinearRegression() # 定义线性回归模型
    # 加载刚刚训练好的模型文件lr.pth
    model.load_state_dict(torch.load('lr.pth'))

    # 将特征向量x输入到model中，得到预测结果h
    h = model(x)

    h_np = h.detach().numpy()
    y_np = y.detach().numpy()
    # 计算h和y之间的MSE、MAE和R2
    mse = mean_squared_error(y_np, h_np)
    mae = mean_absolute_error(y_np, h_np)
    r2 = r2_score(y_np, h_np)
    print(f'Mean Squared Error: {mse:.3f}')
    print(f'Mean Absolute Error: {mae:.3f}')
    print(f'R² Score: {r2:.3f}')


