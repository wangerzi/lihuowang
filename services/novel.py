import json
import os
import random

from services.openai import OpenAIHandler

def split_novel_to_pretrain_data(novel_path: str, target_length: int = 500) -> list:
    """
    将小说内容分割为适合预训练的数据块
    :param novel_path: 小说文件路径
    :param target_length: 目标文本长度，默认500字
    :return: 包含分割后文本的字典列表 [{text: string}]
    """
    try:
        with open(novel_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按行分割并去除空白
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # 合并行到目标长度
        chunks = []
        current_chunk = ""
        
        for line in lines:
            # 如果当前块为空，直接添加
            if not current_chunk:
                current_chunk = line
                continue
            
            # 计算添加当前行后的长度
            potential_length = len(current_chunk) + len(line)
            
            # 如果添加后长度接近目标长度（偏差绝对值最小）
            if abs(potential_length - target_length) < abs(len(current_chunk) - target_length):
                current_chunk += line
            else:
                # 保存当前块并开始新块
                chunks.append({"text": current_chunk})
                current_chunk = line
        
        # 添加最后一个块
        if current_chunk:
            chunks.append({"text": current_chunk})
            
        return chunks
    
    except FileNotFoundError:
        raise Exception("文件未找到")
    except Exception as e:
        raise Exception(f"处理文件时出错: {str(e)}")


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

async def summarize_chapters(chapters: list, openai_service: OpenAIHandler) -> list:
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


async def summarize_qa(chapters: list, openai_service: OpenAIHandler) -> list:
    """
    并行总结小说章节内容，返回包含总结和问答的列表对象
    
    Args:
        chapters: 小说章节内容列表
        openai_service: OpenAI服务实例
        
    Returns:
        list: 包含总结和问答的列表
    """
    from typing import List, Dict, Any
    import asyncio
    
    # 定义验证回调函数
    def validate_response(response: Dict[str, Any]):
        if not isinstance(response, dict):
            raise Exception("Response is not a dictionary")
        if "summary" not in response:
            raise Exception("Missing 'summary' field")
        if "conversations" not in response:
            raise Exception("Missing 'conversations' field")
        if not isinstance(response["conversations"], list):
            raise Exception("'conversations' must be a list")
        for conv_pair in response["conversations"]:
            if not isinstance(conv_pair, list) or len(conv_pair) != 2:
                raise Exception("Conversation pair must be a list of length 2")
            for conv in conv_pair:
                if not isinstance(conv, dict):
                    raise Exception("Conversation item must be a dictionary")
                if "from" not in conv or "value" not in conv:
                    raise Exception("Conversation item missing required fields")
                if conv["from"] not in ["gpt", "human"]:
                    raise Exception("Invalid 'from' value")
            # 验证对话顺序：human -> gpt
            if conv_pair[0]["from"] != "human" or conv_pair[1]["from"] != "gpt":
                raise Exception("Invalid conversation order: must be human -> gpt")

    # 创建任务列表
    tasks = []
    semaphore = asyncio.Semaphore(50)  # 限制并发数
    
    async def process_chapter(index: int, content: str):
        async with semaphore:
            messages = [{
                "role": "system",
                "content": """你是一个专业的小说内容分析专家，请根据小说《道诡异仙》的概括结果和给定待分析的章节内容，首先请总结章节主要内容，然后以多个独立问答形式尽可能全面对该章节剧情、人物、行为、冲突、结果进行描述。
独立问答的意思是：问题中需要自然的带上事件的上下文，比如参与任务、行为、冲突，且一个独立问答只能有一个主要问题，而答案中需要用大白话描述剧情进展、心理变化和结果。为了让更多通过问答看懂剧情，问题不可提及“本章节”、“这一章节”。
《道诡异仙》是一部融合了玄幻、修真、恐怖和心理悬疑元素的小说，主要讲述了主角李火旺在一个诡异而扭曲的世界中挣扎求生的故事。
故事背景
李火旺原本是一个现代社会的普通人，但因为某种原因，他陷入了精神分裂的状态，分不清现实与幻觉。在他的幻觉中，他身处一个充满诡异和恐怖的世界，成为了一个名为“清风观”的道观中的弟子。这个道观由一位名为丹阳子的邪恶道士掌控，丹阳子以炼丹为名，实际上却在用活人炼制丹药，追求成仙之道。
主要情节
清风观的恐怖生活：李火旺在清风观中目睹了丹阳子的残忍行径，丹阳子不仅用活人炼丹，还通过各种邪恶手段控制弟子和药引（被用来炼丹的活人）。李火旺在道观中与其他药引一起生活，目睹了无数人被残忍杀害，内心充满了恐惧和愤怒。
幻觉与现实的交织：李火旺时常在幻觉和现实之间切换，有时他以为自己身处精神病院，有时又回到清风观的恐怖世界。他逐渐意识到，自己可能并不是真的疯了，而是被卷入了一个超自然的世界。
反抗与复仇：李火旺决定反抗丹阳子的暴行，他利用自己的智慧和勇气，逐渐揭开了丹阳子的秘密，并最终设计了一个计划，成功杀死了丹阳子。然而，丹阳子的死亡并没有结束李火旺的苦难，他发现自己仍然被困在这个诡异的世界中。
探索与求生：在丹阳子死后，李火旺带领其他幸存者逃离了清风观，开始了在诡异世界中的探索与求生。他们遇到了各种超自然的存在，如“游老爷”、“黑太岁”等，李火旺逐渐了解到这个世界的真相，并试图找到回到现实世界的方法。
心理挣扎与成长：在整个过程中，李火旺不仅要面对外部的恐怖和危险，还要与自己的内心斗争。他逐渐从一个被动的受害者成长为一个主动的求生者，学会了在这个充满诡异和危险的世界中生存下去。
主题与风格
《道诡异仙》通过李火旺的经历，探讨了现实与幻觉、人性与邪恶、生存与反抗等主题。小说充满了恐怖和悬疑的氛围，情节紧凑，充满了反转和意外。作者通过细腻的心理描写和诡异的世界观构建，成功营造了一个令人毛骨悚然的故事世界。

返回格式要求如下：
1. 对话格式为 JSON 对象，包含 conversations 字段
2. conversations 是一个数组，每个元素是一个对话数组
3. 每个对话数组对象包含 from 和 value 字段
4. from 字段只能是 'gpt' 或 'human'

请严格按照以下JSON格式返回响应：
{
    "summary": "章节主要内容",
    "conversations": [
        [
            {"from": "human", "value": "问题1"},
            {"from": "gpt", "value": "答案1"}
        ],
        [
            {"from": "human", "value": "问题2"},
            {"from": "gpt", "value": "答案2"}
        ]
    ]
}"""
            }, {
                "role": "user",
                "content": f"待分析章节内容：\n{content}"
            }]
            
            response_json = await openai_service.request_json(
                messages=messages,
                temp = 0.7,
                validator_callback=validate_response,
            )
            try:
                response_json["chapter"] = index  # 添加章节索引
                return response_json
            except Exception as e:
                print(f"Error processing chapter {index}: {str(e)}")
                return None

    # 创建所有章节的处理任务
    for index, chapter in enumerate(chapters):
        tasks.append(process_chapter(index, chapter))
        
    # 并行执行所有任务
    results = await asyncio.gather(*tasks)
    # 过滤掉失败的结果
    return [result for result in results if result is not None]

async def summarize_qa_and_save(novel_path: str, conv_output_path: str, summary_output_path: str, openai_service: OpenAIHandler, force: bool = False):
    """
    总结小说内容并保存为两个JSON文件
    :param novel_path: 小说文件路径
    :param conv_output_path: 对话数据集输出路径
    :param summary_output_path: 总结数据集输出路径
    :param openai_service: OpenAI服务实例
    :param force: 是否强制重新生成，即使文件已存在
    """
    try:
        # 如果文件已存在且不强制重新生成，则跳过
        if os.path.exists(conv_output_path) and os.path.exists(summary_output_path) and not force:
            print(f"文件 {conv_output_path} 和 {summary_output_path} 已存在，跳过生成")
            return
        
        # 获取所有章节内容
        chapters = extract_chapters(novel_path)
        
        # 随机选择一个起始索引，确保能取到连续3章
        # start_index = random.randint(0, len(chapters) - 10)
        # 获取连续3章内容
        # summarized_data = await summarize_qa(chapters[start_index:start_index+10], openai_service)
        # 调用总结函数（全量）
        summarized_data = await summarize_qa(chapters, openai_service)

        # 创建datasets目录（如果不存在）
        os.makedirs(os.path.dirname(conv_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)
        
        # 处理对话数据集
        conv_data = []
        for item in summarized_data:
            # 遍历每个章节中的多组对话
            for conv_group in item["conversations"]:
                conv_data.append({
                    "conversations": conv_group,  # 每组对话作为一个独立条目
                    "chapter": item["chapter"]    # 保留章节信息
                })
        
        # 处理总结数据集
        summary_data = []
        for item in summarized_data:
            summary_data.append({
                "summary": item["summary"],
                "chapter": item["chapter"]
            })
        
        # 保存对话数据集
        with open(conv_output_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)
        
        # 保存总结数据集
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"Error: {str(e)}")



