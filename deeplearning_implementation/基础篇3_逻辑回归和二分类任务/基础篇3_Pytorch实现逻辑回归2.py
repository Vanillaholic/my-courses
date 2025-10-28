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

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    # 使用make_blobs随机的生成正例和负例，其中n_samples代表样本数量，设置为50
    # centers代表聚类中心点的个数，可以理解为类别标签的数量，设置为2
    # random_state是随机种子，将其固定为0，这样每次运行就生成相同的数据
    # cluster_s-t-d是每个类别中样本的方差，方差越大说明样本越离散，这里设置为0.5
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)

    posx1, posx2 = X[y == 1][:, 0], X[y == 1][:, 1]
    negx1, negx2 = X[y == 0][:, 0], X[y == 0][:, 1]

    # 创建画板对象，并设置坐标轴
    board = plt.figure()
    axis = board.add_subplot(1, 1, 1)
    # 横轴和纵轴分别对应x1和x2两个特征，长度从-1到6
    axis.set(xlim=[-1, 6],
             ylim=[-1, 6],
             title='Logistic Regression',
             xlabel='x1',
             ylabel='x2')

    # 画出正例和负例，其中正例使用蓝色圆圈表示，负例使用红色叉子表示
    plt.scatter(posx1, posx2, color='blue', marker='o')
    plt.scatter(negx1, negx2, color='red', marker='x')

    # 将数据转化为tensor张量
    X = torch.tensor(X, dtype=torch.float32)
    # view(-1, 1)会将y从1乘50的行向量转为50乘1的列向量
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = LogisticRegression(2)  # 创建逻辑回归模型实例
    criterion = torch.nn.BCELoss()  # 二分类交叉熵损失函数
    # SGD优化器optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(20000):  # 进入逻辑回归模型的循环迭代
        # 将循环的轮数定为2万，就可以使模型达到收敛

        # 在循环中，包括了5个步骤
        h = model(X)  # 1.计算模型的预测值
        loss = criterion(h, y) # 2.计算预测h和标签y之间的损失
        loss.backward()  # 3.使用backward计算梯度
        optimizer.step()  # 4.使用optimizer.step更新参数
        optimizer.zero_grad()  # 5.将梯度清零

        # 这5个步骤，是使用pytorch框架训练模型的定式

        if epoch % 1000 == 0:
            # 每迭代1000次，就打印一次模型的损失，用于观察训练的过程
            # 其中loss.item是损失的标量值
            print(f'After {epoch} iterations, the loss is {loss.item()}')

    # 完成模型的训练后，获取model中的模型参数
    w1 = model.layer.weight[0][0].detach().numpy()
    w2 = model.layer.weight[0][1].detach().numpy()
    b = model.layer.bias[0].detach().numpy()

    # 基于这些参数，通过可视化的方法绘制决策边界
    # 使用linspace在-1到6之间构建间隔相同的100个点
    x = numpy.linspace(-1, 6, 100)
    # 将x代入到分类的决策边界中，计算纵坐标d
    d = - (w1 * x + b) * 1.0 / w2
    plt.plot(x, d) # 进行绘制
    plt.show()


