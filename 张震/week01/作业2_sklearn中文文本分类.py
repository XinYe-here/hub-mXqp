
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import neighbors

# 读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=None)
print(dataset.shape)

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# print(input_sententce)
# print(type(input_sententce))  ## <class 'pandas.core.series.Series'>
# print(input_sententce.values)
vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
# print(input_feature)

knn_model = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_model.fit(input_feature, dataset[1].values)


def featureTextClass(text: str):
    text_sententce = " ".join(jieba.lcut(text))
    text_feature = vector.transform([text_sententce])
    return knn_model.predict(text_feature)


text_arr = ["帮我播放一下郭德纲的小品", ""]
for text in text_arr:
    print(f"{text}的分类是：", featureTextClass(text))



