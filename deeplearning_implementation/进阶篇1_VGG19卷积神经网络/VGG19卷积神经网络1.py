# 预训练的VGG19网络的使用方法
import os
save_path = './pretrained/' # 模型的保存路径
os.environ['TORCH_HOME'] = save_path
os.makedirs(save_path, exist_ok = True)

from torchvision import models
from torchvision.models import VGG19_Weights
# 下载并加载模型
model = models.vgg19(weights = VGG19_Weights.DEFAULT).eval()
print(model)

from PIL import Image
image = Image.open('./data/test.jpg') # 随意找一张测试图片读入
from torchvision import transforms
transform = transforms.Compose([ # 创建转换器
    transforms.Resize(256), # 缩放，调整大小
    transforms.CenterCrop(224), # 中心裁剪
    transforms.ToTensor(), # 转张量
    # 图像标准化处理，其中使用ImageNet数据集的均值和标准差
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# 将该图片转换为合适尺寸的张量
image_tensor = transform(image).unsqueeze(0)
output = model(image_tensor) # 传入model，进行识别
print(output.shape) # 打印output的尺寸

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

# 将imagenet_classes.txt读取
imagenet_id2class = load_imagenet_classes("imagenet_classes.txt")
print(imagenet_id2class) # 打印

import torch
class_id = torch.argmax(output, 1).item() # 找到概率最大的类别
class_name = imagenet_id2class[class_id] # 使用字典，将其转为字符串输出
print(f"{class_id}->{class_name}") # 输出结果


















