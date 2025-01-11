## 简介
源自小说《道诡异仙》，提取火子哥的对话数据集

数据集：
[huggingface 地址](https://huggingface.co/datasets/wj2015/lihuowang-sharegpt)

## 数据清洗
连续对话需要合并同类项，保证基数为 human，偶数为 gpt
如果第一个是 gpt 说的话，前面自动加招呼语 "火旺"、"说话"、"你还好吧" 等招呼语
如果 gpt 中啥也没说，或者最后以 human 结尾，直接给 1 ~ 10 间的 艹

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