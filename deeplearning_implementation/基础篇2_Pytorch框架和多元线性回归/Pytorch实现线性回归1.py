import pandas as pd #用于处理表格

# 读取数据文件，得到data对象
data = pd.read_csv('./data.csv')

# 使用drop函数，删除date、waterfront、view、street、county这5列
data = data.drop(columns=['date', 'waterfront', 'view', 'street', 'country'])

# 因为city和statezip这两列是tag特征
# 所以需要使用cat.codes，将它们转为整数形式
data['city'] = data['city'].astype('category').cat.codes
data['statezip'] = data['statezip'].astype('category').cat.codes

# 获取除了price剩下的12个列的特征
features = data.columns.difference(['price'])
# 对所有特征进行标准化处理
data[features] = (data[features] - data[features].mean()) / data[features].std()

# 将房价price缩小10000倍，使用万作为价格单位
data['price'] = data['price'] / 10000

from sklearn.model_selection import train_test_split

# 使用train_test_split函数
# 将数据data拆分为训练数据train_data和测试数据test_data
train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)

# 使用to_excel将两份数据保存到文件中
train_data.to_excel('./train.xlsx', index=False)
test_data.to_excel('./test.xlsx', index=False)




