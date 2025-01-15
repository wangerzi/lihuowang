## 简介
源自小说《道诡异仙》，使用 DeepSeekv3 提取火子哥的对话数据集和世界观数据集

> deepseek 不愧国产之光，跑完 230w 字的所有章节 SUMMARY 和 QA 就花了 ￥3

## 一些想法

因为小说内容太少，无法支撑继续预训练，除非重头训练大模型并把小说内容注入进去，不过这不太现实。

所以还是要从 SFT 的思路去做，把每个章节的内容都提出尽可能多的，合适的问题，尝试用 SFT 把新知识注入进去

火子哥有点沉默寡言，大多时候内心戏多，说话少，所以对话数据集直接训出来的模型就很呆，也没有故事背景，所以得尝试先有一个道诡异仙世界模型，再继续对火子哥的对话数据集进行清洗优化，才能搞好火子哥的对话模型

..直接 instruct + SFT 质量也不行，这新知识进不了大模型的脑子，大模型缺少这部分知识效果就是不好

数据集：
[huggingface 地址](https://huggingface.co/datasets/wj2015/lihuowang-sharegpt)
[ModelScope 地址](https://www.modelscope.cn/datasets/wangerzi/lihuowang-sharegpt/summary)
[Paddle 飞桨地址](https://aistudio.baidu.com/datasetdetail/313489)

| 数据集名称 | 描述 |
|-----------|------|
| lihuowang-alpaca-dpo.json | 【推荐】使用 Alpaca 格式的 DPO 数据集，包含李火旺的疯言疯语回答和正常回答的对比 |
| daoguiyixian-sharegpt-summary-v2.json | 【推荐】ShareGPT 格式，章节摘要的 QA 第二版，优化了问题质量和回答准确性 |
| daoguiyixian-sharegpt-qa-v2.json | 【推荐】2W 条数据，ShareGPT 格式，针对章节内容的细致问答第二版，增加了更多细节问题和更准确的回答 |
| daoguiyixian-summary-v2.json | 各章节的摘要总结第二版，优化了摘要的连贯性和关键事件的覆盖度 |
| daoguiyixian-sharegpt-summary.json | ShareGPT 格式，章节摘要的 QA |
| daoguiyixian-sharegpt-qa.json | ShareGPT 格式，针对章节内容的细致问答 |
| lihuowang-sharegpt.json | ShareGPT 格式，经过清洗和整理的李火旺对话数据集，包含更规范的对话格式和更完整的对话内容 |
| daoguiyixian-summary.json | 包含各章节的摘要总结，概括了每章的主要情节和关键事件 |
| lihuowang-sharegpt-origin.json | 原始的李火旺对话数据集，包含李火旺与其他角色的对话片段，主要来自小说中的对话内容，有需要可以再自行清洗数据 |

## 进展

#### 2025-01-14

基于 qwen2.5-7b-instruct 5e-5 学习率做两轮 DPO，可以学到疯狂火子哥说话风格

![image](assets/image-20250114180615-e0x9svi.png)

![image](assets/image-20250114180805-nlgua14.png)

![image](assets/image-20250114191407-hqatl7r.png)

![image](assets/image-20250114192320-cvw9t42.png)

不过因为没做 pretrain 和 SFT，DPO 只能学到说话风格，缺少先验知识会有事实性错误

![image](assets/image-20250114191702-r0np6yl.png)

## 数据清洗
连续对话需要合并同类项，保证基数为 human，偶数为 gpt

如果第一个是 gpt 说的话，前面自动加招呼语 "火旺"、"说话"、"你还好吧" 等招呼语

~~如果 gpt 中啥也没说，或者最后以 human 结尾，直接给 1 ~ 10 间的 艹~~

todo:: 如果 gpt 啥也没说，移除掉会不会好点，不然说艹很容易过拟合，如果说话字数少于六个字，是不是也应该移除掉？

## 环境
初始化环境
```bash
conda env create -f environment.yml
```

持久化环境
```bash
conda env export --no-builds > environment.yml
```

## 注意
无授权，不可商用，仅供学习