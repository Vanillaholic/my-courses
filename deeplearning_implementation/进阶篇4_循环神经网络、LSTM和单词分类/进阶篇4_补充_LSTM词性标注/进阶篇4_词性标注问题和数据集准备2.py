from torch.utils.data import Dataset
# 基于dataset封装UDPOS数据集
class POSTagDataset(Dataset):
    # init函数用于初始化，函数传入UDPOS数据
    def __init__(self, data):
        self.examples = list() # 保存词性标注的样本数据
        # 循环读取数据中的每一行
        for i, (text, pos_tags, _) in enumerate(data):
            # 将文本text和标注pos_tag转为小写
            text = [s.lower() for s in text]
            pos_tags = [s.lower() for s in pos_tags]
            # 将文本text和标注pos_tag添加到examples中
            self.examples.append((text, pos_tags))

    def __len__(self):
        return len(self.examples) # 返回列表examples的长度

    def __getitem__(self, index):
        return self.examples[index] # 返回下标为index的数据


# 导入UDPOS数据
from torchtext.datasets import UDPOS

if __name__ == '__main__':
    train_data, _, _ = UDPOS() # 获取训练数据
    # 将train_data传入POSTagDataset，建立数据集dataset
    dataset = POSTagDataset(train_data)
    print(len(dataset)) # 打印数据集的长度
    print(dataset[0][0]) # 打印第1个样本中的文本序列
    print(dataset[0][1]) # 打印第1个样本中的标注序列

