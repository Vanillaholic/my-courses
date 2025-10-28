import matplotlib.pyplot as plt
import numpy as np

# 实现模型的调试代码
if __name__ == '__main__':
    # 定义8个样本，x保存特征值，y保存标记值
    x = np.array([50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    y = np.array([280.0, 305.0, 350.0, 425.0, 480.0, 500.0, 560.0, 630.0])

    # 将样本数据和训练出的直线，进行可视化的输出
    board = plt.figure()  # 创建一个画板
    axis = board.add_subplot(1, 1, 1)  # 创建坐标轴
    axis.set(xlim=[0, 150],  # x轴刻度
             ylim=[0, 800],  # y轴刻度
             title='Linear Regression',  # 坐标系的标题
             xlabel='area',  # x轴标签
             ylabel='price')  # y轴标签
    # 绘制出样本点
    plt.scatter(x, y, color='red', marker='+')

    plt.show()  # 调用show展示结果


















