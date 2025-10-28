import torch
from torch import nn

from models.transformer import Transformer
from tokenizer import Tokenizer
from dataset import TranslateDataset
from loader import collate_batch
from translator import Translator

# 函数传入模型model和src_vocab与trg_vocab两个词表
def test_translate(model, tokenizer):
    sample = "Somebody loves to doll herself up ."  # 定义一个测试样本
    src_tokens = tokenizer.text2tokens(sample)  # 分词结果
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(DEVICE)  # 转为张量

    beam_size = 5
    max_seq_len = 16
    src_pad_idx = tokenizer.get_pad_id()
    trg_pad_idx = tokenizer.get_pad_id()
    trg_bos_idx = tokenizer.get_bos_id()
    trg_eos_idx = tokenizer.get_eos_id()

    translator = Translator(model,
                            beam_size,
                            max_seq_len,
                            src_pad_idx,
                            trg_pad_idx,
                            trg_bos_idx,
                            trg_eos_idx).to(DEVICE)

    translated_indexes = translator.translate_sentence(src_tensor)

    text = tokenizer.tokens2text(translated_indexes)

    print("Somebody loves to doll herself up . -> ", end="")
    print(text)  # 打印出来


from torch.utils.data import DataLoader
from torch import optim
import os


if __name__ == '__main__':
    # 使用TranslateDataset读取训练数据，得到数据集dataset
    # 使用TranslateDataset定义数据集dataset
    t = Tokenizer()
    dataset = TranslateDataset("data/small.txt", t)
    # dataset = TranslateDataset("./data/train.txt")
    print("dataset len:", len(dataset))


    os.makedirs('output', exist_ok=True)  # 建立文件夹，保存迭代过程中的测试图片和模型

    # 定义一个符合DataLoader中collate_fn参数形式的函数collate
    collate = lambda batch: collate_batch(batch, t.get_pad_id())
    # 定义dataloader读取dataset
    dataloader = DataLoader(dataset,
                            batch_size=3,
                            shuffle=False,
                            collate_fn=collate)

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("DEVICE = ", DEVICE)

    # 定义模型的必要参数
    src_vocab_size = t.get_vocab_size()
    trg_vocab_size = t.get_vocab_size()
    src_pad_idx = t.get_pad_id()
    trg_pad_idx = t.get_pad_id()

    #model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(DEVICE)

    # 初始化Transformer模型
    model = Transformer(src_vocab_size=src_vocab_size,
                        src_pad_idx=src_pad_idx,
                        trg_vocab_size=trg_vocab_size,
                        trg_pad_idx=trg_pad_idx,
                        max_len=256, d_model=512, n_head=8,
                        ffn_hidden=2048,
                        n_layers=8,
                        drop_prob=0.1
                        ).to(DEVICE)

    model.train()  # 将model调整为训练模式

    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 定义Adam优化器
    # 定义交叉熵损失函数，需要将<pad>标签设置为ignore_index
    criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

    print("begin train:")
    n_epoch = 50  # 训练轮数设置为50
    for epoch in range(1, n_epoch + 1):  # 外层循环，代表了整个训练数据集的遍历次数
        # 内层循环代表了，在一个epoch中
        # 以随机梯度下降的方式，使用dataloader对于数据进行遍历
        # batch_idx表示当前遍历的批次
        # (text, pos_tag) 表示这个批次的训练数据和词性标记
        for batch_idx, (src, trg) in enumerate(dataloader):  # 遍历dataloader
            # 将src和trg移动到当前设备DEVICE上
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            optimizer.zero_grad()  # 将梯度清零

            # 使用模型model，计算预测结果predict
            predict = model(src, trg[:, :-1])  # 使用模型model计算text的预测结果
            #predict = model(src, trg)

            # 使用view调整predict和标签trg的维度
            predict = predict.view(-1, predict.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)  #右移一位(矫形强制)
            #trg = trg.contiguous().view(-1)

            loss = criterion(predict, trg)  # 计算损失
            loss.backward()  # 计算损失函数关于模型参数的梯度
            # 裁剪梯度，防止梯度爆炸
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()  # 更新模型参数

            # 打印调试信息。包括了当前的迭代轮数epoch
            # 当前的批次batch
            # 当前这个批次的损失loss.item
            print(f"Epoch {epoch}/{n_epoch} "
                  f"| Batch {batch_idx + 1}/{len(dataloader)} "
                  f"| Loss: {loss.item():.4f}")
            # 打印某一个固定样本的翻译效果，观察翻译效果的变化
            
            if (batch_idx + 1) % 20 == 0:
                test_translate(model, t)

        test_translate(model, t)

        """
        model.eval().cpu()
        save_path = f'./output/en2zh_{epoch}.pth'
        torch.save(model.state_dict(), save_path)  # 保存一次模型
        print("Save model: %s" % (save_path))
        model = model.to(DEVICE)
        model.train()
        """

        # 将训练好的模型保存为文件，文件名为translate.model
    torch.save(model.state_dict(), 'output/en2zh.pth')

