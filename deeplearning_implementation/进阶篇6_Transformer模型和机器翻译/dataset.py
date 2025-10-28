from torch.utils.data import Dataset

# 定义TranslateDataset类，读取英译汉的训练数据
class TranslateDataset(Dataset):
    # init函数传入训练数据文件的路径path和分词器tokenizer
    def __init__(self, path, tokenizer):
        file = open(path, 'r', encoding='utf-8')  # 打开训练数据
        self.examples = list()  # 保存英译汉的样本数据
        for line in file:  # 循环读取数据中的每一行
            line = line.strip()
            if not line:
                continue
            # 将每一行，根据\t字符，拆成源语言句子src和目标语言句子trg
            src, trg = line.split('\t')
            # 使用text2tokens对src和trg进行分词
            src_tokens = tokenizer.text2tokens(src)
            trg_tokens = tokenizer.text2tokens(trg)
            # 将分词结果src_tokens和trg_tokens添加到examples
            self.examples.append((src_tokens, trg_tokens))
        file.close()

    def __len__(self):
        return len(self.examples) # 返回列表examples的长度

    def __getitem__(self, index):
        return self.examples[index] # 返回下标为index的数据

from tokenizer import Tokenizer

if __name__ == '__main__':
    t = Tokenizer()
    # 使用TranslateDataset读取small.txt，构造数据集
    dataset = TranslateDataset("./data/small.txt", t)
    # 打印数据集的长度
    print("dataset len:", len(dataset))

    # 接着遍历
    for i in range(len(dataset)):
        src_tokens, trg_tokens = dataset[i]
        print(f"\n样本 {i}:")
        # 打印英译汉样本的分词结果
        print("en tokens:", src_tokens)
        print("zh tokens:", trg_tokens)

        # 使用tokens2text恢复为文本形式
        en_text = t.tokens2text(src_tokens)
        zh_text = t.tokens2text(trg_tokens)
        print("en text:", en_text)
        print("zh text:", zh_text)
