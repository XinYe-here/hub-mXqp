
from openai import OpenAI

client = OpenAI(
    api_key="阿里云百炼的key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def featureTextClass(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {
                "role": "user",
                "content": f"""帮我进行文本分类：{text}
                输出的类型只能从如下中进行选择，除了类别之外下列的类别，请给出最合适的类别。
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
                """
            }
        ]
    )
    return completion.choices[0].message.content

text = "帮我播放一下郭德纲的小品"
print(f"{text}的分类是：", featureTextClass(text))