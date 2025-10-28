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

import torch
def test_sample(model, text_vocab, pos_vocab, text, pos_tag):
    # 实现对一个样本text的预测
    text_tokens = [text_vocab[token[0]] for token in text]
    text_tensor = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(1)
    predict = model(text_tensor)
    max_predict = predict.argmax(-1).squeeze(0)

    itos = pos_vocab.get_itos()
    # 统计这一个样本中正确识别的标签数量、总标签数和正确率
    correct = 0
    tags_num = 0
    for i in range(len(max_predict)):
        predict_label = itos[max_predict[i].item()]
        if predict_label == pos_tag[i][0]:
            correct += 1
        tags_num += 1
    # 正确识别的标签数量、总标签数和正确率
    return correct, tags_num, correct / tags_num

import pickle
from torchtext.datasets import UDPOS
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 读入text_vocab和pos_vocab两个词汇表
    with open("text_vocab.pkl", "rb") as f:
        text_vocab = pickle.load(f)
    with open("pos_vocab.pkl", "rb") as f:
        pos_vocab = pickle.load(f)

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
    # 加载已经训练好的模型
    model.load_state_dict(torch.load('pos_tag.model'))
    model.eval() # 将模型设置为测试模式

    _, _, test_data = UDPOS() # 使用UDPOS获取测试数据
    # 使用POSTagDataset定义数据集dataset
    dataset = POSTagDataset(test_data)
    # 定义dataloader读取dataset
    test_loader = DataLoader(dataset)

    print("total test num: %d"%(len(dataset)))
    all_correct = 0 # 正确标签的总数
    all_tags_num = 0 # 标签总数
    # 遍历全部的测试样本
    for idx, (text, tags) in enumerate(test_loader):
        # 对于每个测试样本，调用函数test_sample测试效果
        correct, tags_num, acc = test_sample(model, text_vocab, pos_vocab, text, tags)
        # 打印测试结果
        print("%d: %d/%d = %.3lf"%(idx, correct, tags_num, acc))
        all_correct += correct
        all_tags_num += tags_num
    acc = all_correct / all_tags_num
    # 打印总的测试效果
    print("accuracy: %d/%d = %.3lf"%(all_correct, all_tags_num, acc))

