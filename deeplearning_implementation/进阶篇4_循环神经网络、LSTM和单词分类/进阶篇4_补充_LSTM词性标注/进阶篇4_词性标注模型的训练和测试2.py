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

from torch import nn
# 实现一个用于解决词性标注的BiLSTM模型

class BiLSTMPOSTagger(nn.Module):
    # input_dim是输入数据的维度，对应文本词汇表中单词的个数
    # embedding_dim，词向量的维度
    # hidden_dim隐藏层神经元个数
    # output_dim是输出数据的维度，对应标注词汇表中标注的个数
    # n_layers是隐藏层的层数
    # dropout是丢弃比率
    def __init__(self, input_dim, embedding_dim, hidden_dim,
                 output_dim, n_layers, dropout):
        super().__init__()
        # 定义词嵌入层
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size = embedding_dim, # 输入维度
                            hidden_size = hidden_dim, # 隐藏层维度
                            num_layers = n_layers, # 隐藏层的数量
                            bidirectional = True, # 双向结构
                            # 当隐藏层数大于1层时，设置dropout丢弃比率
                            dropout = dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout) # dropout层
        # 由于是双向LSTM，因此fc的输入大小是hidden_dim*2，输出大小是output_dim
        self.fc = nn.Linear(hidden_dim * 2 , output_dim)

    # 在前向传播forward函数中，函数传入文本序列text
    def forward(self, text):
        x = self.embedding(text) # 将文本序列输入至词嵌入层
        x = self.dropout(x) # 将x输入至dropout层
        output, (_, _) = self.lstm(x) # 将x输入至lstm层
        output = self.dropout(output) # 输入至dropout层
        predict = self.fc(output) # 输入至fc层，计算出词性标注结果
        return predict # 返回predict

# 函数传入预测结果predict，标签label和标签中的填充索引pad_idx
def accuracy(predict, label, pad_idx):
    correct = 0 # 正确的标签数量
    total = 0 # 总标签数量
    for i in range(len(label)): # 遍历所有的标签
        if label[i] == pad_idx:
            # 忽略掉其中的填充标签
            continue
        # 计算预测结果中最大的标签
        max_predict = predict[i].argmax(dim = 0)
        if max_predict == label[i]:
            correct += 1 # 正确的数量加1
        total += 1 # 总数加1
    return correct / total # 返回正确率

import torch
from torchtext.datasets import UDPOS
from torch.utils.data import DataLoader
from torch import optim
import pickle

if __name__ == '__main__':
    train_data, _, _ = UDPOS() # 使用UDPOS获取训练数据
    dataset = POSTagDataset(train_data)  # 使用POSTagDataset定义数据集
    # 使用build_vocab，建立文本词表和标注词表
    text_vocab, pos_vocab = build_vocab(dataset)
    # 打印两个词表长度
    print("text_vocab:", len(text_vocab))
    print("pos_vocab:", len(pos_vocab))
    # 将两个词表保存下来，词表也相当于模型的一部分
    with open("text_vocab.pkl", "wb") as f:
        pickle.dump(text_vocab, f)
    with open("pos_vocab.pkl", "wb") as f:
        pickle.dump(pos_vocab, f)

    # 定义一个符合DataLoader中collate_fn参数形式的函数collate
    collate = lambda batch: collate_batch(batch, text_vocab, pos_vocab)
    # 定义dataloader读取dataset
    dataloader = DataLoader(dataset,
                            batch_size = 32, # 每个小批量包含32个数据
                            shuffle = True, # 将数据打乱顺序后读取
                            collate_fn = collate)
                            
    # 定义模型的必要参数
    INPUT_DIM = len(text_vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(pos_vocab)
    N_LAYERS = 2
    DROPOUT = 0.25
    # 定义模型
    model = BiLSTMPOSTagger(INPUT_DIM,
                            EMBEDDING_DIM,
                            HIDDEN_DIM,
                            OUTPUT_DIM,
                            N_LAYERS,
                            DROPOUT)
    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("DEVICE = ", DEVICE)
    # 将model转到DEVICE上
    model = model.to(DEVICE)
    model.train() # 将model调整为训练模式
    optimizer = optim.Adam(model.parameters()) # 定义Adam优化器
    #定义交叉熵损失函数，需要将<pad>标签设置为ignore_index
    criterion = nn.CrossEntropyLoss(ignore_index = pos_vocab["<pad>"])

    print("begin train:")
    n_epoch = 30 # 训练轮数设置为30
    for epoch in range(n_epoch):  # 外层循环，代表了整个训练数据集的遍历次数
        loss_sum = 0  # 用于打印调试信息

        # 内层循环代表了，在一个epoch中
        # 以随机梯度下降的方式，使用dataloader对于数据进行遍历
        # batch_idx表示当前遍历的批次
        # (text, pos_tag) 表示这个批次的训练数据和词性标记
        for batch_idx, (text, pos_tag) in enumerate(dataloader): #遍历dataloader
            # 将text和postag移动到当前设备DEVICE上
            text = text.to(DEVICE)
            pos_tag = pos_tag.to(DEVICE)
            optimizer.zero_grad() # 将梯度清零

            predict = model(text) # 使用模型model计算text的预测结果
            # 使用view调整predict和标签pos_tag的维度
            # [seq_length, batch_size, output_dim]
            # -> [seq_length * batch_size, output_dim]
            predict = predict.view(-1, predict.shape[-1])
            # [seq_length, batch_size] -> [seq_length * batch_size]
            pos_tag = pos_tag.view(-1)

            loss = criterion(predict, pos_tag) # 计算损失
            loss.backward() # 计算损失函数关于模型参数的梯度
            optimizer.step()  # 更新模型参数

            loss_sum += loss.item()  # 累加当前样本的损失
            # 每训练100个批次，打印一次调试信息
            if (batch_idx + 1) % 100 == 0:
                # 计算当前这一批次的正确率
                acc = accuracy(predict, pos_tag, pos_vocab["<pad>"])
                print(f"Epoch {epoch + 1}/{n_epoch} " # 当前的迭代轮数
                    f"| Batch {batch_idx + 1}/{len(dataloader)} " # 当前的批次
                    f"| Loss: {loss_sum:.4f}" # 当前这100组数据的累加损失
                    f"| acc: {acc:.4f}") # 当前批次的正确率
                loss_sum = 0

    # 将训练好的模型保存为文件，文件名为pos_tag.model
    torch.save(model.state_dict(), 'pos_tag.model')


