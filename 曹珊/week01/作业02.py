import jieba
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI


def data_preprocessing():
    # 读取数据,pandas 用于进行表格的加载和分析
    data = pd.read_csv("dataset.csv", sep='\t', header=None, nrows=10000)
    # 对文本进行分词
    input_sententce = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
    # 对文本进行特征提取
    vector = CountVectorizer()
    # 统计词表
    vector.fit(input_sententce.values)
    # 向量化
    input_feature = vector.transform(input_sententce.values)
    return input_feature, data[1].values, vector


def model_by_KNN(input_feature, label):
    # 训练模型，让模型学习句子与标签的关系
    model = KNeighborsClassifier()
    model.fit(input_feature, label)
    return model


def model_by_linear_model(input_feature, lable):
    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(input_feature, lable)
    return model


def text_calssify_using_ml(text: str, model, vector) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]


def text_calssify_using_llm(text: str) -> str:
    """
    文本分类(大语言模型)，输入文本完成类别划分
    """
    client = OpenAI(
        api_key="sk-8f965d790ae24d5188af80effe79a180",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role":"user", "content":f"""
帮我进行文本分类：{text}，输出的类别只能从如下选择，只输出类别，不要输出其他的内容：
FilmTele-Play          
Video-Play               
Music-Play              
Radio-Listen            
Alarm-Update             
Travel-Query             
HomeAppliance-Control    
Weather-Query            
Calendar-Query           
TVProgram-Play            
Audio-Play                
Other"""}
        ]
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    # 提取的数据特征和标签
    input_feature, label, vector = data_preprocessing()
    # 导入模型训练
    model_knn = model_by_KNN(input_feature, label)
    # 测试
    print(text_calssify_using_ml("帮我导航到天安门", model_knn, vector))

    model_lm = model_by_linear_model(input_feature, label)
    print(text_calssify_using_ml("帮我导航到南大门", model_lm, vector))

    print(text_calssify_using_llm("帮我导航到外婆家"))


