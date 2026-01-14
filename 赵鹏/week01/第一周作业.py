'''

# 导包
import jieba
import pandas as pd
STOPWORDS = ["的", "吗", "了", "一下", "一个", "吧", "这", "那"]  # 停用词列表
CATEGORY_KEYWORDS = {
    "Travel-Query": ["车票", "飞机", "高铁", "回家", "自驾", "路线", "导航", "大巴", "航班", "火车"],
    "Music-Play": ["播放", "歌", "音乐", "单曲循环", "专辑", "钢琴曲", "歌曲", "演唱会"],
    "FilmTele-Play": ["电视剧", "电影", "剧", "花絮", "古装剧", "偶像剧", "电影预告"],
    "Video-Play": ["游戏视频", "比赛视频", "直播", "动漫", "解说", "综艺", "新闻视频"],
    "Radio-Listen": ["广播", "电台", "FM", "收听", "电台节目", "交通广播", "音乐广播"],
    "HomeAppliance-Control": ["空调", "冰箱", "洗衣机", "烤箱", "加湿器", "风速", "温度", "开关"],
    "Alarm-Update": ["提醒", "闹钟", "取消提醒", "定闹钟", "日程", "打卡", "签到"],
    "Weather-Query": ["温度", "下雨", "刮风", "紫外线", "湿度", "天气", "降雨", "下雪", "冰雹"],
    "Calendar-Query": ["农历", "星期几", "几号", "节日", "元旦", "春节", "端午节"],
    "Audio-Play": ["有声小说", "故事", "广播剧", "评书", "小说连播"],
    "TVProgram-Play": ["电视节目", "春晚", "CCTV", "卫视"],
    "Other": []
}
# 1、读取csv文件
df = pd.read_csv('dataset.csv', sep='\t', header=None, names=['text', 'label'])
# df.info()

# 统计各类别数量
count_result = df["label"].value_counts()
# print(f"数据集中各类别数量：{count_result}")

# 2、数据预处理
df['text'] = df['text'].fillna('')
df['label'] = df['label'].fillna('Other')
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

# 3、定义分词函数（精确模式，去除停用词）
def cut_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ""  
    words = jieba.lcut(text)  
    filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    return " ".join(filtered_words)

# 4、对文本列分词
df["cut_words"] = df["text"].apply(cut_text)

# 5、定义基于分词的分类函数（优化关键词匹配逻辑）
def classify_by_cut_words(cut_words_str):
    if not isinstance(cut_words_str, str) or cut_words_str.strip() == '':
        return "Other"  

    cut_words = cut_words_str.split()
    for label, keywords in CATEGORY_KEYWORDS.items():
        if label == "Other":
            continue
        for word in cut_words:
            if any(keyword in word for keyword in keywords):
                return label
    return "Other"


# 6、对分词后的文本分类
df["predicted_label"] = df["cut_words"].apply(classify_by_cut_words)

# 7、查看分类结果
# print("\n分类结果示例（前5行）：")
# print(df[["text", "cut_words", "label", "predicted_label"]].head())

# 8、测试输入文字判断类型
def predict_single_text():
    print("\n===== 文本分类测试器 =====")
    print("输入任意文本，程序会输出对应的label；输入 'exit' 退出测试")
    while True:
        user_input = input("\n请输入要测试的文本：").strip()
        if user_input.lower() == 'exit':
            print("测试结束！")
            break
        if not user_input:
            print("输入不能为空，请重新输入！")
            continue
        cut_result = cut_text(user_input)
        predicted_label = classify_by_cut_words(cut_result)
        print(f" 输入文本：{user_input}")
        print(f"分词结果：{cut_result if cut_result else '无有效分词'}")
        print(f"预测标签：{predicted_label}")
        
if __name__ == "__main__":
    predict_single_text()



'''
