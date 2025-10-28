import tiktoken

class Tokenizer():
    # 初始化tiktoken分词器
    def __init__(self):
        # 定义特殊的单词索引
        # 这些索引值不会与tiktoken分词器中的其他单词索引值有冲突
        self.BOS = 100300 # 起始索引
        self.EOS = 100301 # 结束索引
        self.PAD = 100302 # 填充索引
        self.MAX_ID = 100500 # 最大索引

        # 定义特殊单词到特殊单词索引的字典
        # 字符串形式的单词到整数的索引映射关系
        special_tokens = {
            "<bos>": self.BOS,
            "<eos>": self.EOS,
            "<pad>": self.PAD,
        }

        # 获得一个编码解码器，t是一个基础的编码解码器
        # 用于将文本编码为tokens_id或者将tokens_id解码为文本
        t = tiktoken.get_encoding("cl100k_base")

        # 将刚刚预定义的特殊单词索引special_tokens添加到t中
        # 构造带有特殊单词索引的编码解码器
        self.t = tiktoken.Encoding(
            name="cl100k_base_with_special",
            pat_str=t._pat_str,
            mergeable_ranks=t._mergeable_ranks,
            special_tokens={**t._special_tokens, **special_tokens}
        )

    # 用于编码，即将传入的文本text转换为单词的整数索引序列
    def text2tokens(self, text):
        return [self.BOS] + self.t.encode(text) + [self.EOS]

    # 用于解码，即将单词的整数索引序列恢复为文本序列
    def tokens2text(self, tokens):
        # 去掉头和尾添加的BOS和EOS
        return self.t.decode(tokens[1:-1])

    # 在训练英译汉Transformer模型的过程中
    # 还需要获取特殊单词对应的索引
    def get_bos_id(self):
        return self.BOS # 获取起始符号BOS

    def get_eos_id(self):
        return self.EOS # 获取结束符号EOS

    def get_pad_id(self):
        return self.PAD # 获取填充符号PAD

    def get_vocab_size(self):
        return self.MAX_ID # 获取词表大小

if __name__ == "__main__":
    tokenizer = Tokenizer() # 创建分词器
    text = "你好吗？" # 测试文本
    # 将文本转为单词索引
    tokens = tokenizer.text2tokens(text)
    # 将单词索引恢复为文本
    decoded_text = tokenizer.tokens2text(tokens)

    # 打印这些变量
    print("原始文本:", text)
    print("单词索引:", tokens)
    print("解码后的文本:", decoded_text)
    print("BOS_ID:", tokenizer.get_bos_id())
    print("EOS_ID:", tokenizer.get_eos_id())
    print("PAD_ID:", tokenizer.get_pad_id())
    print("词表大小:", tokenizer.get_vocab_size())
