# 使用其中的UDPOS数据集，做词性标注训练
# UDPOS数据集可以直接在torchtext.dataset中导入
from torchtext.datasets import UDPOS

# UDPOS数据集包括了三个子集合
# 分别是训练集train_data、验证集valid_data和测试集test_data
train_data, valid_data, test_data = UDPOS()

# 打印它们的长度
print(f"train_data len: {len(list(train_data))}")
print(f"valid_data len: {len(list(valid_data))}")
print(f"test_data len: {len(list(test_data))}")

# 遍历训练数据，从中取出文本text和词性标注pos_tag
for i, (text, pos_tags, _) in enumerate(train_data):
    # 打印其中的前三个数据
    if i >= 3:
        break
    print(f"Sample {i+1}:")
    print(f"Text: {text}")
    print(f"POS Tags: {pos_tags}")


