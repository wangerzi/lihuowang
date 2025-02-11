import random
from datasets import Dataset
import os
from dotenv import load_dotenv
import asyncio
import json
from services.novel import split_novel_to_pretrain_data, lihuowang_sharegpt_and_save, summarize_qa_and_save
from services.openai import OpenAIHandler

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

async def convert_sharegpt_to_alpaca(sharegpt_path: str, alpaca_path: str, instruct: str) -> None:
    """将sharegpt格式数据转换为alpaca格式
    
    参数:
        sharegpt_path: sharegpt格式数据文件路径
        alpaca_path: 输出alpaca格式数据文件路径
        instruct: 指令模板
    """
    # 读取sharegpt数据
    with open(sharegpt_path, "r", encoding="utf-8") as f:
        sharegpt_data = json.load(f)
    
    alpaca_data = []
    for item in sharegpt_data:
        # 确保对话格式正确
        if len(item["conversations"]) < 2:
            continue
        if item["conversations"][0]["from"] != "human" or item["conversations"][1]["from"] != "gpt":
            continue
            
        # 构建alpaca格式
        alpaca_item = {
            "instruction": instruct,
            "input": item["conversations"][0]["value"],
            "output": item["conversations"][1]["value"]
        }
        alpaca_data.append(alpaca_item)
    
    # 保存转换后的数据
    with open(alpaca_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)


async def convert_summary_to_sharegpt(summary_path, output_path):
    """将章节摘要转换为sharegpt格式"""
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    
    # 问题模板
    question_templates = [
        "《道诡异仙》第{chapter}章主要讲了什么内容",
        "《道诡异仙》第{chapter}章主要内容是什么",
        "《道诡异仙》第{chapter}章主要写了啥",
        "《道诡异仙》第{chapter}章讲了什么",
        "《道诡异仙》第{chapter}章主要是什么剧情",
        "《道诡异仙》第{chapter}章剧情是什么",
        "《道诡异仙》第{chapter}章内容是什么",
        "《道诡异仙》第{chapter}章他们做了什么事",
        "《道诡异仙》第{chapter}章他们干了什么事"
    ]
    
    sharegpt_data = []
    for item in summary_data:
        chapter = item["chapter"] + 1  # 章节号+1
        question = random.choice(question_templates).format(chapter=chapter)
        
        sharegpt_data.append({
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": item["summary"]}
            ],
            "chapter": item["chapter"]
        })
    
    # 保存转换后的数据
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

async def main():
    # 加载环境变量
    load_dotenv()

    # 调用分割函数生成预训练数据
    pretrain_data = split_novel_to_pretrain_data("./novel.txt")
    
    # 创建datasets目录（如果不存在）
    os.makedirs("datasets", exist_ok=True)
    
    # 保存预训练数据
    with open("datasets/daoguiyixian-pretrain.json", "w", encoding="utf-8") as f:
        json.dump(pretrain_data, f, ensure_ascii=False, indent=2)
    
    # 初始化openai服务
    openai_service = OpenAIHandler(
        openai_key=os.getenv("OPENAI_API_KEY"),
        openai_url=os.getenv("OPENAI_BASE_URL"),
        model="deepseek-chat"
    )
    
    
    # 调用QA总结函数
    await summarize_qa_and_save(
        novel_path="./novel.txt",
        conv_output_path="datasets/daoguiyixian-sharegpt-qa-v2.json",
        summary_output_path="datasets/daoguiyixian-summary-v2.json",
        openai_service=openai_service,
        # force=True
    )
    
    # 将摘要转换为sharegpt格式
    await convert_summary_to_sharegpt(
        summary_path="datasets/daoguiyixian-summary-v2.json",
        output_path="datasets/daoguiyixian-sharegpt-summary-v2.json"
    )
    # 将sharegpt格式的摘要数据转换为alpaca格式
    await convert_sharegpt_to_alpaca(
        sharegpt_path="datasets/daoguiyixian-sharegpt-summary-v2.json",
        alpaca_path="datasets/daoguiyixian-alpaca-summary-v2.json",
        instruct="请用你理解的《道诡异仙》小说内容解答用户疑惑"
    )
    
    # 将sharegpt格式的QA数据转换为alpaca格式
    await convert_sharegpt_to_alpaca(
        sharegpt_path="datasets/daoguiyixian-sharegpt-qa-v2.json",
        alpaca_path="datasets/daoguiyixian-alpaca-qa-v2.json",
        instruct="请用你理解的《道诡异仙》小说内容解答用户疑惑"
    )
    
    # 调用封装后的函数
    await lihuowang_sharegpt_and_save(
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
