import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN

dataset = pd.read_csv("../data/dataset.csv", sep="\t", header=None, nrows=100)
print(dataset.head(5))

# jieba 对中文进行处理
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector = CountVectorizer()
# 统计词表
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
# 第一种方式：
# 创建knn模型
model = KNeighborsClassifier()
# 进行模型训练
model.fit(input_feature, dataset[1].values)
#查看model
print(model)

# 模型测试
test_query = "请帮我查出近期河北各省气温总指数"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))

# 第二种方式：
from sklearn.naive_bayes import MultinomialNB
# 创建朴素贝叶斯分类器
model = MultinomialNB()
# 进行模型训练
model.fit(input_feature, dataset[1].values)
#查看model
print(model)

# 模型测试
test_query = "请帮我查出近期河北各省气温总指数"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))
