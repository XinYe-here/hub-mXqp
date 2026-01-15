import sys
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI
sys.stdout.reconfigure(encoding="utf-8")
from fastapi import FastAPI
import os
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

load_dotenv()
app = FastAPI()

## 数据准备
dir_path = os.path.dirname(os.path.realpath(__file__))
print("根目录是：",dir_path)
data_name = "dataset.csv"
dataset_path = os.path.join(dir_path,"..",data_name)
dataset = pd.read_csv(dataset_path, sep="\t", header=None, nrows=30000)
print(dataset[1].value_counts())# 查看种类

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer() # 看看所有文本里出现过哪些词，并给每个词一个编号
vector.fit(input_sentence.values) # 扫描所有文本，建立词表（vocabulary）
input_feature = vector.transform(input_sentence.values) # 按刚才学到的词典，把每一句话转成词频向量

model_tree = DecisionTreeClassifier(
    max_depth=30,          # 防止树太深过拟合
    min_samples_leaf=2,    # 叶子至少2个样本
    random_state=42
)
model_tree.fit(input_feature, dataset[1].values)

model_logistic = LogisticRegression(
    max_iter=2000,     # 文本特征多，迭代次数要大
)
model_logistic.fit(input_feature, dataset[1].values)

client = OpenAI(
    api_key= os.getenv("DASHSCOPE_API_KEY"),

    # 大模型厂商的地址
    base_url= os.getenv("DASHSCOPE_BASE_URL")
)
def text_classify_using_ml(text: str, model) -> str:
    """
    文本分类（机器学习）
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0] # 返回里面的str

def text_classify_using_llm(text: str) -> str:
    """
    文本分类 (大语言模型)，输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME"), 
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text} 输出的类别只能从如下中进行选择，除了类别之外不要有其他内容，请给出最合适的类别。{list(dataset[1].value_counts().index)}"""},
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    print(text_classify_using_ml("今天天气不错,我准备去打球", model_tree)) # Weather-Query
    print(text_classify_using_ml("今天天气不错,我准备去打球", model_logistic)) # Music-Play
    print(text_classify_using_llm("今天天气不错,我准备去打球")) # Weather-Query 大模型对照组