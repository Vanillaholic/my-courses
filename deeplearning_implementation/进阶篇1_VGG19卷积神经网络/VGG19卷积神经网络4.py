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

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os

if __name__ == '__main__':
    transform = transforms.Compose([  # 创建转换器
        transforms.Resize(256),  # 缩放，调整大小
        # 使用了数据增强技术:
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转，15是旋转角度，范围是[-15, 15]
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转张量
        # 图像标准化处理，其中使用ImageNet数据集的均值和标准差
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # 将训练数据的目录，传入ImageFolder，构造数据集
    train_data = datasets.ImageFolder("./flowers_data/train",
                                      transform)
    print(f"traind_data size = {len(train_data)}") # 打印train_data的长度

    class_num = len(train_data.classes) # 获取类别数量
    label2index = train_data.class_to_idx # 字符串标签到数字
    index2label = {idx: label for label, idx in label2index.items()} # 数字到字符串标签
    print("class_num: ", class_num) # 打印类别数量
    print("label2index: ", len(label2index)) # 打印字典长度
    print(label2index) # 打印字典
    print("index2label: ", len(index2label)) # 打印字典长度
    print(index2label) # 打印字典

    train_load = DataLoader(train_data, # 使用DataLoader读取数据集
                            batch_size = 32,
                            shuffle = True)
    print(f"train_load size = {len(train_load)}") # 打印DataLoader的长度

    # 读取数据后，定义设备对象device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = %s"%(device))

    model_path = './pretrained/hub/checkpoints/vgg19-dcbb9e9d.pth'
    vgg19 = FlowerVGG19(class_num, model_path).to(device) # 定义花卉模型
    vgg19.train() # 调整为训练模式

    # 定义Adam优化器，这里要注意，我们要优化的是分类层中的参数
    optimizer = optim.Adam(vgg19.model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 创建一个交叉熵损失函数

    os.makedirs('models', exist_ok = True)

    for epoch in range(1, 51): # 进入50轮的循环迭代
        for batch_idx, (data, label) in enumerate(train_load):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()  # 清空梯度

            output = vgg19(data) # 前向传播，计算output
            loss = criterion(output, label) # 计算损失
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型参数

            # 每组数据迭代，都计算这组数据的正确率，用于观察训练效果
            predict = torch.argmax(output, dim=1)
            correct = (predict == label).sum().item()
            acc = correct / data.size(0)

            print(f"Epoch {epoch}/50 " # 迭代轮数
                  f"| Batch {batch_idx + 1}/{len(train_load)} " # 批次编号
                  f"| Loss: {loss.item():.4f}" # 当前批次的损失
                  f"| acc: {correct}/{data.size(0)}={acc:.3f}") # 当前批次的正确率

        if epoch % 5 == 0: # 每训练5个epoch
            model_name = f'./models/flowers_epoch{epoch}.pth'
            print("saved model: %s"%(model_name))
            torch.save(vgg19.state_dict(), model_name) # 保存一次模型

