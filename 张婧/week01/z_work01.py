import pandas as pd  # 提取csv文档数据，用来进行表格的加载和分析
import jieba  # 中文分词
from sklearn.feature_extraction.text import CountVectorizer  # 统计词频
from sklearn.neighbors import KNeighborsClassifier  # KNN
from openai import OpenAI

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)  # 读取数据
print(dataset[1].value_counts())  # 输出dataset的类型列并统计次数

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 批量处理分词列表
vector = CountVectorizer()  # 对文本进行特征提取
vector.fit(input_sententce.values)  # 统计词表
input_feature = vector.transform(input_sententce.values)  # 进行转换

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)


client = OpenAI(
    # 绑定百炼API_Key
    api_key="sk-14fe5xxxxxxxbf8b1ef0e0",

    # 阿里云大模型地址，
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 使用的模型名

        messages=[
            {"role": "user", "content": f"""进行文本分类：{text}   

输出的类别只能从如下中进行选择，请给出最合适的类别。
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
"""},  # 模拟用户进行提问
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":

    print("机器学习: ", text_calssify_using_ml("五一劳动节是农历几时"))
    print("大语言模型: ", text_calssify_using_llm("播放一首适合开心时候听的歌曲呢"))
