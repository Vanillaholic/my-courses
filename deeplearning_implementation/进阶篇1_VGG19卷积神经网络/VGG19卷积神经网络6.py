import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGG19, self).__init__()
        # 定义features，它是一个Sequential容器模块
        self.features = nn.Sequential(
            # 第1组卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第2组卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第3组卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第4组卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第5组卷积块
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # 自适应平均池化层
        self.classifier = nn.Sequential( # 分类器
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x): # 前向传播函数
        x = self.features(x) # 经过特征层
        x = self.avgpool(x) # 平均池化层
        x = torch.flatten(x, 1) # 将张量x展平
        x = self.classifier(x) # 分类器计算结果
        return x

class FlowerVGG19(nn.Module):
    def __init__(self, num_classes, model_path = None):
        super(FlowerVGG19, self).__init__()
        self.model = VGG19() # 定义原始的VGG19模型
        if model_path: # 如果model_path不为空
            # 此时要训练模型，所以需要加载预训练的原始VGG19模型
            weights = torch.load(model_path)
            self.model.load_state_dict(weights)
            for param in self.model.features.parameters():
                # 将model的features中的全部参数固定住
                param.requires_grad = False
        # 获取最后一个线性层的输入特征数
        num_features = self.model.classifier[-1].in_features
        # 构造新的线性层
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)
    def forward(self, x):
        return self.model(x) # 直接调用model

from PIL import Image
from torchvision import transforms
import os

if __name__ == '__main__':
    transform = transforms.Compose([  # 创建转换器
        transforms.Resize(256),  # 缩放，调整大小
        # 在测试的transform中，不要添加旋转等数据增强操作:
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.RandomRotation(15),  # 随机旋转，15是旋转角度，范围是[-15, 15]
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转张量
        # 图像标准化处理，其中使用ImageNet数据集的均值和标准差
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    index2label = {0: 'astilbe',
                   1: 'bellflower',
                   2: 'black_eyed_susan',
                   3: 'calendula',
                   4: 'california_poppy',
                   5: 'carnation',
                   6: 'common_daisy',
                   7: 'coreopsis',
                   8: 'daffodil',
                   9: 'dandelion',
                   10: 'iris',
                   11: 'magnolia',
                   12: 'none',
                   13: 'rose',
                   14: 'sunflower',
                   15: 'tulip',
                   16: 'water_lily'}

    # 加载模型并设置为评估模式
    vgg19 = FlowerVGG19(len(index2label))
    # 加载训练好的模型
    vgg19.load_state_dict(torch.load('flowers_epoch50.pth'))
    vgg19.eval()

    check_dir = "./check/"  # 待检查的目录
    files = os.listdir(check_dir)
    for file_name in files:
        file_path = os.path.join(check_dir, file_name)
        image = Image.open(file_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        output = vgg19(image)
        predict = output.argmax(1).item()
        print(f"{file_path}\t->\t{index2label[predict]}")
