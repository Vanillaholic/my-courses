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

import torch
from torch.nn.utils.rnn import pad_sequence

# 函数collate_batch，对于每个小批量batch，进行填充操作
# 另外还需要传入文本词表text_vocab和标记词表pos_vocab
def collate_batch(batch, text_vocab, pos_vocab):
    text = list()
    tags = list()
    for text_sample, tags_sample in batch: # 遍历batch中的样本
        # 将文本序列text_sample和标记序列tags_sample
        # 通过词表，转换为索引序列
        text_tokens = [text_vocab[token] for token in text_sample]
        tag_tokens = [pos_vocab[token] for token in tags_sample]
        # 将它们添加到列表text和tags中
        text.append(torch.tensor(text_tokens, dtype=torch.long))
        tags.append(torch.tensor(tag_tokens, dtype=torch.long))
    # 使用pad_sequence，对text和tags填充
    text = pad_sequence(text, padding_value = text_vocab["<pad>"])
    tags = pad_sequence(tags, padding_value = pos_vocab["<pad>"])
    return text, tags # 返回text和tags

from torch.utils.data import DataLoader
from torchtext.datasets import UDPOS

if __name__ == '__main__':
    train_data, _, _ = UDPOS()
    dataset = POSTagDataset(train_data) # 创建数据集dataset
    text_vocab, pos_vocab = build_vocab(dataset) # 创建两个词表

    # 创建lambda函数collate，它是一个符合DataLoader中collate_fn参数形式的函数
    # 由于collate_batch需要传入词汇表text_vocab和pos_vocab
    # 它不符合collate_fn参数的形式，因此需要使用lambda函数做包装处理
    collate = lambda batch: collate_batch(batch, text_vocab, pos_vocab)
    # 接着定义dataloader读取dataset
    dataloader = DataLoader(dataset,
                            batch_size = 4,
                            shuffle = False,
                            collate_fn = collate)

    for batch_idx, (text, pos_tag) in enumerate(dataloader): #遍历dataloader
        if batch_idx >= 3:
            break
        # 打印每个小批次
        print("batch_idx:", batch_idx)
        print("text:")
        print(text)
        print("pos_tag:")
        print(pos_tag)


