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

# 函数传入映射文件文件的路径path，函数会输出映射字典id2class
def load_imagenet_classes(path):
    id2class = {} # 映射字典
    # 字典的key是类别的整数id，字典的value是类别的名称
    # 例如，数字1就对应了“goldfish”
    lines = open(path, 'r').readlines()
    for line in lines:
        key, value = line.strip().split(', ')
        id2class[int(key)] = value
    return id2class #返回映射字典

from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    # 使用torch.load加载下载的预训练模型
    weights = torch.load('./pretrained/hub/checkpoints/vgg19-dcbb9e9d.pth')
    model = VGG19()
    model.load_state_dict(weights) # 加载参数
    model.eval()

    image = Image.open('./data/test.jpg') # 随意找一张测试图片读入
    transform = transforms.Compose([  # 创建转换器
        transforms.Resize(256),  # 缩放，调整大小
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转张量
        # 图像标准化处理，其中使用ImageNet数据集的均值和标准差
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 将该图片转换为合适尺寸的张量
    image_tensor = transform(image).unsqueeze(0)
    output = model(image_tensor)  # 传入model，进行识别
    print(output.shape)  # 打印output的尺寸

    # 将imagenet_classes.txt读取
    imagenet_id2class = load_imagenet_classes("imagenet_classes.txt")
    print(imagenet_id2class)  # 打印

    class_id = torch.argmax(output, 1).item()  # 找到概率最大的类别
    class_name = imagenet_id2class[class_id]  # 使用字典，将其转为字符串输出
    print(f"{class_id}->{class_name}")  # 输出结果




