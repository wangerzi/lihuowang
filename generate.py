import random
from datasets import Dataset
import os
from dotenv import load_dotenv
import asyncio
import json
from services.openai import OpenAIHandler


def extract_chapters(novel_path):
    """
    从小说文件中提取各章内容
    :param novel_path: 小说文件路径
    :return: 包含各章内容的字符串数组
    """
    try:
        with open(novel_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则表达式匹配章节标题并分割内容
        import re
        chapters = re.split(r'第\d+章.*\n', content)
        
        # 去除第一个空元素（标题前的内容）
        if chapters and not chapters[0].strip():
            chapters = chapters[1:]
            
        return chapters
    except FileNotFoundError:
        raise Exception("File not found")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

async def summarize_chapters(chapters: list, openai_service) -> list:
    """
    并行总结小说章节内容，返回sharegpt格式的列表对象
    
    Args:
        chapters: 小说章节内容列表
        openai_service: OpenAI服务实例
        
    Returns:
        list: sharegpt格式的对话列表
    """
    from typing import List, Dict, Any
    import asyncio
    
    # 定义验证回调函数
    def validate_response(response: Dict[str, Any]):
        if not isinstance(response, dict):
            raise Exception("Response is not a dictionary")
        if "conversations" not in response:
            raise Exception("Missing 'conversations' field")
        if not isinstance(response["conversations"], list):
            raise Exception("'conversations' must be a list")
        for conv in response["conversations"]:
            if not isinstance(conv, dict):
                raise Exception("Conversation item must be a dictionary")
            if "talk" not in conv:
                raise Exception("Missing 'talk' field")
            if not isinstance(conv["talk"], list):
                raise Exception("'talk' must be a list")
            for talk in conv["talk"]:
                if not isinstance(talk, dict):
                    raise Exception("Talk item must be a dictionary")
                if "from" not in talk or "value" not in talk:
                    raise Exception("Talk item missing required fields")
                if talk["from"] not in ["gpt", "human"]:
                    raise Exception("Invalid 'from' value")

    # 创建任务列表
    tasks = []
    semaphore = asyncio.Semaphore(50)  # 限制并发数
    
    async def process_chapter(index: int, content: str):
        async with semaphore:
            messages = [{
                "role": "system",
                "content": """你是一个专业的小说对话总结助手。请将小说《道诡异仙》的章节内容总结为主角李火旺的多段对话

返回格式要求如下：
1. 对话格式为 JSON 对象，包含 conversations 字段
2. conversations 是一个数组，每个元素是一个对话对象
3. 每个对话对象包含 talk 字段，talk 是一个数组
4. 每个 talk 对象包含 from 和 value 字段
5. from 字段只能是 'gpt' 或 'human'
6. 有可能通篇李火旺都没有说话，只有其他角色的对话或心理描写，这时 conversations 请为空


注意：对话中一定要包含 human 和 gpt，不能只有一个人在说

对话内容的要求如下：
1. gpt 是指主角李火旺说出来的话，在书中可能被称做 李师兄、红中、火旺、化名耳玖等
2. human 是指其他角色说出来的话，或以说话的方式描述情况
3. 对话内容要忠实原文，保持人物关系和情感
4. 对话要突出关键情节和人物互动
5. 对话要自然流畅，符合人物性格
6. 主角李火旺的精神在大傩世界和现实世界来回穿梭，注意那些疯言疯语和语气助词 艹


角色关系参考如下：
现代世界：
父:李建成 母:孙晓琴
女朋友:杨娜
医生:王韦、易东来、吴成
合作者:清旺来、钱福、陈红瑜、赵雷、赵霜点、 巴楠旭、巴晟清、五琦

大傩世界：
师弟妹:狗娃(曹操)、白灵淼(妻子、白莲圣女)、赵五、高志坚(大梁皇帝)、春小满、杨小孩（胥民）
妻子:白灵淼（二神）
女儿:李岁(玄牝)
徒弟:吕秀才
友人:诸葛渊


请严格按照以下JSON格式返回响应：
{
  "conversations": [
    {
      "talk": [
        {
          "from": "human",
          "value": "其他人说的话"
        },
        {
          "from": "gpt", 
          "value": "主角李火旺说的话"
        }
      ]
    }
  ]
}"""
            }, {
                "role": "user",
                "content": content,
            }]
            try:
                response = await openai_service.request_json(
                    messages=messages,
                    temp = 0,
                    validator_callback=validate_response,
                )
                response["capter"] = index  # 添加章节索引
                return response
            except Exception as e:
                print(f"Error processing chapter {index}: {str(e)}")
                return {"conversations": [], "capter": index}
    
    # 启动所有任务
    for index, chapter in enumerate(chapters):
        tasks.append(process_chapter(index, chapter))
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    # 合并结果，以talk为维度组织conversations
    final_result = []
    for result in results:
        for conv in result["conversations"]:
            final_result.append({
                "conversations": conv['talk'],  # 直接将talk作为conversations的内容
                "capter": result["capter"]
            })
    
    return final_result

async def summarize_and_save(novel_path: str, output_path: str, openai_service: OpenAIHandler, force: bool = False):
    """
    总结小说内容并保存为JSON文件
    :param novel_path: 小说文件路径
    :param output_path: 输出JSON文件路径
    :param openai_service: OpenAI服务实例
    :param force: 是否强制重新生成，即使文件已存在
    """
    try:
        # 如果文件已存在且不强制重新生成，则跳过
        if os.path.exists(output_path) and not force:
            print(f"文件 {output_path} 已存在，跳过生成")
            return
        
        # 获取所有章节内容
        chapters = extract_chapters(novel_path)
        
        # 调用总结函数
        # 随机选择一个起始索引，确保能取到连续3章
        # start_index = random.randint(0, len(chapters) - 20)
        # 获取连续3章内容
        # selected_chapters = chapters[start_index:start_index+20]
        # summarized_chapters = await summarize_chapters(selected_chapters, openai_service)
        # 取第一章测试
        # summarized_chapters = await summarize_chapters(chapters[:1], openai_service)
        # 跑全量
        summarized_chapters = await summarize_chapters(chapters, openai_service)

        # 创建datasets目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 将结果保存为JSON文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summarized_chapters, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"Error: {str(e)}")

async def clean_dataset(data):
    """
    清理数据集
    :param data: 原始数据集
    :return: 清理后的数据集
    """
    import random
    cleaned_data = []
    
    # 定义随机内容
    random_human = ["火旺", "说话", "你还好吧", "醒醒", "李火旺！！"]
    random_gpt = lambda: "艹" * random.randint(1, 10)
    
    for item in data:
        try:
            # 合并连续相同from的内容
            merged_conversations = []
            prev_from = None
            for conv in item["conversations"]:
                if conv["from"] == prev_from:
                    merged_conversations[-1]["value"] += " " + conv["value"]
                else:
                    merged_conversations.append(conv)
                prev_from = conv["from"]
            
            # 过滤空值并处理
            filtered_conversations = []
            for conv in merged_conversations:
                if conv["from"] == "human" and not conv["value"].strip():
                    continue
                if conv["from"] == "gpt" and not conv["value"].strip():
                    conv["value"] = random_gpt()
                filtered_conversations.append(conv)
            
            # 处理开头和结尾
            if filtered_conversations and filtered_conversations[0]["from"] == "gpt":
                filtered_conversations.insert(0, {
                    "from": "human",
                    "value": random.choice(random_human)
                })
            
            # 如果结尾是human，移除最后一条human
            if filtered_conversations and filtered_conversations[-1]["from"] == "human":
                filtered_conversations.pop()
                
            # 如果移除后只剩下human，跳过这段对话
            if all(conv["from"] == "human" for conv in filtered_conversations):
                continue
            
            # 检查对话有效性
            if not filtered_conversations:
                print(f"Warning: Empty conversations in item {item}")
                continue
                
            if (len(filtered_conversations) < 2 or 
                filtered_conversations[0]["from"] != "human" or
                filtered_conversations[1]["from"] != "gpt" or
                filtered_conversations[-1]["from"] != "gpt"):
                print(f"Warning: Invalid conversation structure in item {item}")
                continue
                
            # 保存有效数据
            cleaned_data.append({
                "conversations": filtered_conversations,
                "capter": item["capter"]
            })
            
        except Exception as e:
            print(f"Error processing item {item}: {str(e)}")
    
    print(f"Final dataset size: {len(cleaned_data)}")
    return cleaned_data

async def main():
    # 加载环境变量
    load_dotenv()
    
    # 初始化openai服务
    openai_service = OpenAIHandler(
        openai_key=os.getenv("OPENAI_API_KEY"),
        openai_url=os.getenv("OPENAI_BASE_URL"),
        model="deepseek-chat"
    )
    
    # 调用封装后的函数
    await summarize_and_save(
        novel_path="./novel.txt",
        output_path="datasets/lihuowang-sharegpt-origin.json",
        openai_service=openai_service
    )
    
    # 读取原始数据
    with open("datasets/lihuowang-sharegpt-origin.json", "r", encoding="utf-8") as f:
        origin_data = json.load(f)
    
    # 清理数据
    cleaned_data = await clean_dataset(origin_data)

    # 保存清理后的数据为json文件
    with open("datasets/lihuowang-sharegpt.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    # 使用datasets库保存清理后的数据
    dataset = Dataset.from_dict({
        "conversations": [item["conversations"] for item in cleaned_data],
        "capter": [item["capter"] for item in cleaned_data]
    })
    
    # 保存数据集
    dataset.save_to_disk("datasets/lihuowang-sharegpt")

if __name__ == "__main__":
    try:
        asyncio.run(main=main())
    except Exception as e:
        print(f"Error: {str(e)}")
