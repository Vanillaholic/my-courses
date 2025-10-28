from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

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



