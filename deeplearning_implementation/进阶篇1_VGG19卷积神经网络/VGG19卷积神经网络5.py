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
from torchvision import transforms

if __name__ == '__main__':
    transform = transforms.Compose([  # 创建转换器
        transforms.Resize(256),  # 缩放，调整大小
        # 在测试的transform中，不要添加旋转等数据增强操作:
        #transforms.RandomHorizontalFlip(),  # 随机水平翻转
        #transforms.RandomRotation(15),  # 随机旋转，15是旋转角度，范围是[-15, 15]
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转张量
        # 图像标准化处理，其中使用ImageNet数据集的均值和标准差
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 将训练数据的目录，传入ImageFolder，构造数据集
    test_data = datasets.ImageFolder("./flowers_data/test",
                                      transform)
    print(f"test_data size = {len(test_data)}")  # 打印train_data的长度

    class_num = len(test_data.classes)  # 获取类别数量
    label2index = test_data.class_to_idx  # 字符串标签到数字
    index2label = {idx: label for label, idx in label2index.items()}  # 数字到字符串标签
    print("class_num: ", class_num)  # 打印类别数量
    print("label2index: ", len(label2index))  # 打印字典长度
    print(label2index)  # 打印字典
    print("index2label: ", len(index2label))  # 打印字典长度
    print(index2label)  # 打印字典

    # 读取数据后，定义设备对象device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = %s" % (device))

    vgg19 = FlowerVGG19(class_num).to(device)
    # 加载训练好的模型
    vgg19.load_state_dict(torch.load('flowers_epoch50.pth'))
    vgg19.eval()

    correct = 0
    total = 0
    for i, (data, label) in enumerate(test_data): # 遍历全部的测试集
        data = data.unsqueeze(0).to(device)
        label = torch.tensor([label], dtype=torch.long).to(device)
        predict = vgg19(data).argmax(1)
        # 测试效果
        if predict.eq(label).item() == True:
            correct += 1
        else:
            predict_str = index2label[predict.item()]
            file_path, _ = test_data.imgs[i]
            # 将所有识别错的图像路径打出来，用于观察调试
            print(f"wrong case: {file_path}\t->\t{predict_str}")
        total += 1

    acc = 100 * correct / total
    print(f'Accuracy: {correct}/{total}={acc:.3f}%')



