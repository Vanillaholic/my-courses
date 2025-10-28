import torch
from models.transformer import Transformer
from tokenizer import Tokenizer
from translator import Translator

if __name__ == '__main__':
    t = Tokenizer()

    src_vocab_size = t.get_vocab_size()
    trg_vocab_size = t.get_vocab_size()
    src_pad_idx = t.get_pad_id()
    trg_pad_idx = t.get_pad_id()
    trg_bos_idx = t.get_bos_id()
    trg_eos_idx = t.get_eos_id()

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("DEVICE = ", DEVICE)

    model = Transformer(src_vocab_size=src_vocab_size, src_pad_idx=src_pad_idx,
                              trg_vocab_size=trg_vocab_size, trg_pad_idx=trg_pad_idx,
                              max_len=256, d_model=512, n_head=8,
                              ffn_hidden=2048, n_layers=8, drop_prob=0.1).to(DEVICE)

    model.load_state_dict(torch.load('./output/en2zh.pth'))
    model.eval()

    beam_size = 5
    max_seq_len = 16

    translator = Translator(model,
                            beam_size,
                            max_seq_len,
                            src_pad_idx,
                            trg_pad_idx,
                            trg_bos_idx,
                            trg_eos_idx).to(DEVICE)

    sample = "I am a student ."  # 定义一个测试样本
    src_tokens = t.text2tokens(sample)  # 分词结果
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(DEVICE)  # 转为张量

    translated_indexes = translator.translate_sentence(src_tensor)
    predict_word = t.tokens2text(translated_indexes)
    print("I am a student . -> ", end="")
    print(predict_word)  # 打印出来
