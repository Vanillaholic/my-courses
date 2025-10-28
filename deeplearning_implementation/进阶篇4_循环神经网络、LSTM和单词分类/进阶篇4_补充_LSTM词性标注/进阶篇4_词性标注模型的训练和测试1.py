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

if __name__ == '__main__':
    INPUT_DIM = 8866 # 有8866个文本词汇
    EMBEDDING_DIM = 100 # 词向量维度是100
    HIDDEN_DIM = 128 # 隐藏层神经元个数
    OUTPUT_DIM = 19 # 输出层维度，表示有19个标注词
    N_LAYERS = 2 # 隐藏层数量
    DROPOUT = 0.25 # 丢弃比率
    # 定义模型
    model = BiLSTMPOSTagger(INPUT_DIM,
                            EMBEDDING_DIM,
                            HIDDEN_DIM,
                            OUTPUT_DIM,
                            N_LAYERS,
                            DROPOUT)
    # 打印模型，可以看到BiLSTMPOSTagger模型的结构
    print(model)

