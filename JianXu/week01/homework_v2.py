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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import sklearn

# annotation
ModelType = DecisionTreeClassifier | LogisticRegression


# -------- logging --------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
logger = logging.getLogger("text-cls")

load_dotenv()
app = FastAPI()

_is_inited = False

def init_models() -> None:
    global _is_inited, labels, X_test, y_test, model_tree, model_logistic, vector
    if _is_inited:
        return

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.normpath(os.path.join(dir_path, "..", "dataset.csv"))
    dataset = pd.read_csv(dataset_path, sep="\t", header=None, nrows=30000)
    labels = list(dataset[1].value_counts().index)
    input_sentence = dataset[0].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))

    X_text = dataset[0].astype(str)
    y = dataset[1].values 

    X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y,  # X, y
    test_size=0.2, 
    random_state=42,
    stratify=dataset[1].values # 保证训练集和测试集的类别比例一致
    )

    X_train_sentence = X_train_text.apply(lambda x: " ".join(jieba.lcut(x)))
    X_test_sentence = X_test_text.apply(lambda x: " ".join(jieba.lcut(x)))
    vector = CountVectorizer() #
    vector.fit(X_train_sentence.values) # 在分词前就需要将train test分离，避免偷看答案
    X_train = vector.transform(X_train_sentence.values)
    X_test = vector.transform(X_test_sentence.values)
    

    model_tree = DecisionTreeClassifier(
        max_depth=30,          
        min_samples_leaf=2,    
        random_state=42
    )
    model_logistic = LogisticRegression(
        max_iter=2000,     
    )

    model_tree.fit(X_train, y_train)
    model_logistic.fit(X_train, y_train)

    # pred_tree = model_tree.predict(X_test)
    # pred_logistic = model_logistic.predict(X_test)
    # tree_acc = accuracy_score(y_test, pred_tree)
    # logistic_acc = accuracy_score(y_test, pred_logistic)
    # print("Tree acc：",tree_acc)
    # print("LR acc：",logistic_acc)
    # logger.info(f"Tree acc： {tree_acc:.4f}")
    # logger.info(f"LR acc： {logistic_acc:.4f}")
    _is_inited = True
    return

def evaluate_model(model: ModelType, x, y) -> float:
    """
    评估模型，返回准确率
    """
    pred = model.predict(x)
    acc = accuracy_score(y, pred)
    model_name = type(model).__name__
    logger.info(f"{model_name} accuracy: {acc:.4f}")
    return accuracy_score(y_test, pred)


def text_classify_using_ml(text: str, model: ModelType) -> str:
    """
    文本分类 (机器学习模型)，输入文本完成类别划分
    """
    text = text.strip()
    if not text:
        logger.warning("Invalid empty text.")
        return "INVALID_EMPTY_TEXT"
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = vector.transform([text_sentence])
    return model.predict(text_feature)[0]

def init_llm_model() -> None:
    """
    初始化大模型
    """
    global client, llm_model_name
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "qwen-flash")

    if not api_key:
        logger.warning("Missing DASHSCOPE_API_KEY in environment (.env).")
        raise RuntimeError("Missing DASHSCOPE_API_KEY in environment (.env).")

    client = OpenAI(api_key=api_key, base_url=base_url)

def text_classify_using_llm(text: str) -> str:

    """
    文本分类 (大语言模型)，输入文本完成类别划分
    """
    text = text.strip()
    if not text:
        logger.warning("Invalid empty text.")
        return "INVALID_EMPTY_TEXT"
    completion = client.chat.completions.create(
        model=llm_model_name, 
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text} 输出的类别只能从如下中进行选择，除了类别之外不要有其他内容，请给出最合适的类别。{labels}"""},
        ],
    )
    return completion.choices[0].message.content

def test_models(input_text: str) -> str:
    """
    测试模型，输入文本完成类别划分,对比结果
    """
    output_tree = text_classify_using_ml(input_text, model_tree)
    output_logistic = text_classify_using_ml(input_text, model_logistic)
    output_llm = text_classify_using_llm(input_text)
    
    return f"Tree: {output_tree}, Logistic: {output_logistic}, LLM: {output_llm}"


if __name__ == "__main__":
    init_models()
    init_llm_model()
    evaluate_model(model_tree, X_test, y_test)
    evaluate_model(model_logistic, X_test, y_test)
    print(test_models("今天天气真好,我们去打羽毛球吧"))


'''
代码总结：
1.使用初始化方法，将模型初始化放在了主函数中，这样在调用函数时，可以避免重复初始化，提高代码的效率。模型的初始化就做单一的准备，不要添加其他的逻辑。
2. 对于初始化的模型，使用了全局变量来保存，这样可以在函数中直接使用，而不需要每次都传入模型。
3. API等配置信息，使用了环境变量来保存，这样可以在不同的环境中使用不同的配置，提高了代码的灵活性。
4. 对于初始化的模型使用，注意异常处理
5. 使用logging 模块来记录日志，方便后续的调试和查看。
6. 在文本分类中，向量器必须只在训练集上 fit，并在测试集上仅使用 transform；测试集中未见过的词会被忽略，这是防止数据泄露、衡量模型真实泛化能力的必要设计。
'''