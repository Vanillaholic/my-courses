import numpy as np

# J(θ)关于theta0的偏导数计算函数
# 函数传入样本特征x和标记值y，参数theta0和theta1
def gradient_theta0(x, y, theta0, theta1):
    m = len(x)  # 保存样本的个数
    h = theta0 + theta1 * x # 预测值
    # 计算预测值和真实值的差，调用sum函数求和，再除以m
    return np.sum(h - y) / m

# 实现J(θ)关于theta1的偏导数计算函数
# 函数同样传入x、y，theta0和theta1
def gradient_theta1(x, y, theta0, theta1):
    m = len(x)  # 样本个数
    h = theta0 + theta1 * x # 预测值
    # 计算预测值和真实值的差，额外乘一个特征x
    # 然后调用sum函数求和，再除以m
    return np.sum((h - y) * x) / m

# 实现代价函数的计算函数
# 函数不会影响梯度下降的过程，只是用来调试与观察结果
def costJ(x, y, theta0, theta1):
    m = len(x)  # 保存样本的个数
    h = theta0 + theta1 * x # 计算预测值
    # 根据均方误差公式，计算每个样本误差的平方
    # 使用sum累加到一起，再除以2m
    return np.sum((h - y) ** 2) / (2 * m)

# 梯度下降的迭代函数
# 函数传入特征x、标记值y、模型迭代速率alpha和迭代次数n
def gradient_descent(x, y, alpha, n):
    theta0 = 0.0 # 初始化
    theta1 = 0.0
    for i in range(1, n+1):  # 梯度下降的迭代循环，循环n次
        # 计算theta0和theta1的偏导数g0和g1
        g0 = gradient_theta0(x, y, theta0, theta1)
        g1 = gradient_theta1(x, y, theta0, theta1)
        # 提前计算g0和g1，然后再更新theta0和theta1
        # 可以保证同时更新这两个参数
        theta0 = theta0 - alpha * g0 # 梯度下降更新g0
        theta1 = theta1 - alpha * g1 # 梯度下降更新g1
    return theta0, theta1  # 函数返回theta0和theta1

import matplotlib.pyplot as plt

# 实现模型的调试代码
if __name__ == '__main__':
    # 定义8个样本，x保存特征值，y保存标记值
    x = np.array([50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    y = np.array([280.0, 305.0, 350.0, 425.0, 480.0, 500.0, 560.0, 630.0])
    alpha = 0.0001  # 迭代速率
    n = 100  # 迭代次数
    # 调用梯度下降函数，计算迭代结果，对应预测直线的截距和斜率
    theta0, theta1 = gradient_descent(x, y, alpha, n)
    cost = costJ(x, y, theta0, theta1) # 计算代价函数
    print("After %d iterations, the cost is %lf" % (n, cost))
    print("theta0 = %lf theta1 = %lf" % (theta0, theta1))
    # 使用预测函数predict，预测面积为112和110的房价
    print("predict(112) = %lf" % (theta0 + theta1 * 112))
    print("predict(110) = %lf" % (theta0 + theta1 * 110))

    # 将样本数据和训练出的直线，进行可视化的输出
    board = plt.figure()  # 创建一个画板
    axis = board.add_subplot(1, 1, 1) # 创建坐标轴
    axis.set(xlim=[0, 150], # x轴刻度
             ylim=[0, 800], # y轴刻度
             title='Linear Regression', # 坐标系的标题
             xlabel='area', # x轴标签
             ylabel='price') # y轴标签
    # 绘制出样本点
    plt.scatter(x, y, color='red', marker='+')
    # 在0到150之间，构造出500个相同间距的浮点数，保存至x
    x = np.linspace(0, 150, 500)
    h = theta1 * x + theta0  # 计算直线的预测值
    plt.plot(x, h)  # 画出直线
    plt.show()  # 调用show展示结果
