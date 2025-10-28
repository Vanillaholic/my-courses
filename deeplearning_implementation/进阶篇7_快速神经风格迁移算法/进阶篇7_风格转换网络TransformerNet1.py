from torch import nn

class ResidualBlock(nn.Module): # 定义残差块
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 第1个卷积层序列
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1), # 边缘反射填充
            nn.Conv2d(channels, channels, kernel_size=3, stride=1), # 卷积层
            nn.InstanceNorm2d(num_features=channels, affine=True), # 实例归一化
            nn.ReLU() # 激活函数
        )
        # 第2个卷积层序列
        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(1), # 边缘反射填充
            nn.Conv2d(channels, channels, kernel_size=3, stride=1), # 卷积层
            nn.InstanceNorm2d(num_features=channels, affine=True) # 实例归一化
        )

    def forward(self, x):
        fx = self.layer1(x) # 输入x先通过layer1
        fx = self.layer2(fx) # 再通过layer2，得到主路径的计算结果
        return x + fx # 返回原始输入x与主路径计算结果fx的和

import torch

if __name__ == "__main__":
    # 定义一个5*128*64*64的输入数据
    # 其中5表示样本个数，128表示输入通道数量，64*64是特征图大小
    x = torch.randn(5, 128, 64, 64)
    model = ResidualBlock(128) # 输入的特征数为128，与输入数据的输入通道数相同
    output = model(x) # 计算前向传播
    print(f"output.shape = {output.shape}") # 打印输出结果的尺寸


















