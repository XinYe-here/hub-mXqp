# 机器学习路线：文本→分词→词频向量→KNN预测-->我搜索了朴素贝叶斯模型
# 大模型路线：文本→提示词→模型直接输出类别
import jieba
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.naive_bayes import MultinomialNB # 朴素贝叶斯
from openai import OpenAI


# 使用pandas读取csv文件
data_csv = pd.read_csv("dataset.csv",sep='\t',header=None,nrows=None)
input_sententce = data_csv[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理，取csv文件的第一列进行分析处理

vector = CountVectorizer() # 对文本进行提取特征
vector.fit(input_sententce.values) # 统计出现过哪些词
input_feature = vector.transform(input_sententce.values) # 把每句话转换成数字，例如有 100 条句子，就会得到 100 行向量

# 把knn模型改为朴素贝叶斯模型
model = MultinomialNB()
model.fit(input_feature, data_csv[1].values)

# 机器学习

def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

# 配置openai数据
client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"), # 配置了环境变量
    # 阿里云地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""对我的文本进行分类：{text}
输出的类别只能以下类别选择，若没有匹配度高的请输出other。
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
Other
"""},
        ]
    )
    return completion.choices[0].message.content



if __name__ == "__main__":

    print("机器学习: ", text_calssify_using_ml("成都在哪里"))
    print("大语言模型",text_calssify_using_llm("明天温度是多少"))
