# 预训练的VGG16网络的使用方法
import os
save_path = './pretrained/' # 模型的保存路径
os.environ['TORCH_HOME'] = save_path
os.makedirs(save_path, exist_ok = True)

from torchvision import models
from torchvision.models import VGG16_Weights
# 下载并加载模型
model = models.vgg16(weights = VGG16_Weights.DEFAULT).eval()
print(model)

from PIL import Image
image = Image.open('./data/airplane.jpg') # 随意找一张测试图片读入
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

# 后续我们会专门使用VGG16的特征提取层
# 也就是前16个卷积层，对风格图像和内容图像进行特征提取
model = models.vgg16(weights = VGG16_Weights.DEFAULT).eval().features
output = model(image_tensor) # 计算图像的特征
print(output.shape) # 打印output的尺寸



