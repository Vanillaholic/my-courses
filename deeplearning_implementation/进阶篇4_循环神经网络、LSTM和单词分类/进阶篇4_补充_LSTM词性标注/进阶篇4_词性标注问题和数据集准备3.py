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

from torchtext.vocab import build_vocab_from_iterator
# 实现build_vocab函数，基于数据集dataset，建立词汇表
# 词汇表包括两个，分别是文本词汇表和标注词汇表
def build_vocab(dataset):
    # unk表示未知词，pad表示填充词
    special = ["<unk>", "<pad>"]
    text_iter = map(lambda x: x[0], dataset) # 文本序列
    pos_iter = map(lambda x: x[1], dataset) # 标注序列
    # 建立文本词汇表text_vocab和目标语言词汇表pos_vocab
    # 将min_freq设置为2，也就是至少出现两次的单词，才会添加到词表中
    text_vocab = build_vocab_from_iterator(text_iter, min_freq = 2, specials = special)
    pos_vocab = build_vocab_from_iterator(pos_iter, min_freq = 2, specials = special)
    # 将unk对应的索引，设置为默认索引
    text_vocab.set_default_index(text_vocab["<unk>"])
    pos_vocab.set_default_index(pos_vocab["<unk>"])
    return text_vocab, pos_vocab # 返回两个词汇表

from torchtext.datasets import UDPOS

if __name__ == "__main__":
    train_data, _, _ = UDPOS()
    # 使用POSTagDataset定义数据集dataset
    dataset = POSTagDataset(train_data)
    # 使用build_vocab，建立文本词表和标注词表
    text_vocab, pos_vocab = build_vocab(dataset)
    # 打印两个词表长度
    print("text_vocab:", len(text_vocab))
    print("pos_vocab:", len(pos_vocab))
    # 打印pos_vocab.get_itos，可以得到全部的词性标注单词
    print(pos_vocab.get_itos())



















