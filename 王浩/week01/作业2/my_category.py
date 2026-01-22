import pandas as pd
import jieba
import os
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("../dataset.csv", sep="\t", header=None)
input_sen = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
vector = CountVectorizer()
vector.fit(input_sen.values)
input_feature = vector.transform(input_sen.values)

model = KNeighborsClassifier()
model.fit(input_feature, data[1].values)

# 使用sklearn
def text_classify_sklearn(text: str) -> str:
    test_sen = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sen])
    return model.predict(test_feature)

# 调用大模型获取结果
def text_classify_llm(text:str) -> str:
    my_api_key = os.getenv("API_KEY")
    my_url = os.getenv("BASE_URL")
    my_model = os.getenv("MODEL")
    if not my_api_key:
        print("api key为空!")
        return ""
    if not my_url:
        print("base url为空!")
        return ""
    if not my_model:
        print("model为空!")
        return ""
    client = OpenAI(api_key= my_api_key, base_url=my_url)
    classify_categories = data[1].dropna().unique().tolist()
    content = f"""帮我对以下文本进行分类：
    文本内容：{text}

    输出要求：
    1. 只能从指定类别中选择一个作为结果；
    2. 指定类别列表：{', '.join(classify_categories)}；
    3. 仅返回分类结果，不要额外解释。
    """

    completion = client.chat.completions.create(
        model=my_model,
        messages=[
            {"role": "user", "content": content},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    test_query = "帮我导航到天安门"
    classify01 = text_classify_sklearn(test_query)
    classify02 = text_classify_llm(test_query)
    print("类型: ", classify01)
    print("大模型-类型: ", classify02)