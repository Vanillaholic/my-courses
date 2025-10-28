import torch
from torch.nn.utils.rnn import pad_sequence
from dataset import TranslateDataset

# 函数collate_batch，对于每个小批量batch，进行填充操作
# 另外还需要传入源语言词表src_vocab和目标语言词表trg_vocab
def collate_batch(batch, pad_id):
    src = list()
    trg = list()
    for src_tokens, trg_tokens in batch:  # 遍历batch中的样本
        # 将它们添加到列表src和trg中
        src.append(torch.tensor(src_tokens, dtype=torch.long))
        trg.append(torch.tensor(trg_tokens, dtype=torch.long))
    # 使用pad_sequence，对src和trg填充
    src = pad_sequence(src, padding_value=pad_id).transpose(0, 1)
    trg = pad_sequence(trg, padding_value=pad_id).transpose(0, 1)
    return src, trg  # 返回src和trg

from torch.utils.data import DataLoader
from tokenizer import Tokenizer

if __name__ == '__main__':
    # 使用TranslateDataset定义数据集dataset
    t = Tokenizer()
    dataset = TranslateDataset("data/small.txt", t)
    print("dataset len:", len(dataset))

    # 创建lambda函数collate，它是一个符合DataLoader中collate_fn参数形式的函数
    # 由于collate_batch需要传入词汇表src_vocab和trg_vocab
    # 它不符合collate_fn参数的形式，因此需要使用lambda函数做包装处理
    collate = lambda batch: collate_batch(batch, t.get_pad_id())
    # 接着定义dataloader读取dataset
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            shuffle=False,
                            collate_fn=collate)

    for batch_idx, data in enumerate(dataloader):  # 遍历dataloader
        src = data[0]
        trg = data[1]
        # 打印每个小批次
        print("batch_idx:", batch_idx)
        print("src:", src.shape)
        print(src)
        print("trg:", trg.shape)
        print(trg)
