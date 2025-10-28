import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def hypothesis(theta, x):
    # NumPy 的 dot 函数直接计算点积，这里的 x 和 theta 都可以是数组
    return sigmoid(np.dot(x, theta))

def gradients(x, y, theta):
    m = len(y)  # 样本数量
    h = hypothesis(theta, x)  # 计算所有样本的预测值
    grads = np.dot(x.T, h - y) / m  # 通过矩阵乘法计算所有梯度
    return grads

def gradient_descent(x, y, alpha, iterate):
    m, n = x.shape  # m是样本数，n是特征数（已经包括了截距项）
    theta = np.zeros(n)  # 初始化参数
    for _ in range(iterate):
        grads = gradients(x, y, theta)
        theta -= alpha * grads  # 更新所有参数
    return theta

# 代价函数
def costJ(x, y, theta):
    m = len(y)
    h = hypothesis(theta, x)
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m

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
    # 横轴和纵轴分别对应x1和x2两个特征，长度从-1到5，画板名称为SVM
    axis.set(xlim=[-1, 6],
             ylim=[-1, 6],
             title='Logistic Regression',
             xlabel='x1',
             ylabel='x2')

    # 画出正例和负例，其中正例使用蓝色圆圈表示，负例使用红色叉子表示
    plt.scatter(posx1, posx2, color='blue', marker='o')
    plt.scatter(negx1, negx2, color='red', marker='x')

    m = len(X)  # 保存样本个数
    n = 2  # 保存特征个数
    alpha = 0.001  # 迭代速率
    iterate = 20000  # 迭代次数
    # 将生成的特征向量X的添加一列1，作为偏移特征
    X = numpy.insert(X, 0, values=[1] * m, axis=1)

    # 调用梯度下降算法，迭代出分界平面，并计算代价值
    theta = gradient_descent(X, y, alpha, iterate)
    costJ = costJ(X, y, theta)
    for i in range(0, len(theta)):
        print("theta[%d] = %lf" % (i, theta[i]))
    print("Cost J is %lf" % (costJ))

    # 根据迭代出的模型参数，绘制分类的决策边界
    w1 = theta[1]
    w2 = theta[2]
    b = theta[0]
    # 使用linspace在-1到5之间构建间隔相同的100个点
    x = numpy.linspace(-1, 6, 100)
    # 将这100个点，代入到决策边界，计算纵坐标
    d = - (w1 * x + b) * 1.0 / w2
    # 绘制分类的决策边界
    plt.plot(x, d)
    plt.show()