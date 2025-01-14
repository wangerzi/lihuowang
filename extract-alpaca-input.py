import argparse
from datasets import load_dataset

def extract_input_to_file(dataset_name, dataset_key, output_file):
    """
    从指定数据集中提取input字段，并用换行符分隔保存到目标文件
    :param dataset_name: 数据集名称
    :param dataset_key: 数据集键名（如train, test等）
    :param output_file: 输出文件名
    """
    try:
        # 加载数据集
        dataset = load_dataset(dataset_name)
        
        # 检查指定的key是否存在
        if dataset_key not in dataset:
            raise ValueError(f"Dataset key '{dataset_key}' not found in dataset")
        
        # 提取input字段
        inputs = [item['input'] for item in dataset[dataset_key]]
        
        # 用换行符连接所有input
        content = "\n".join(inputs)
        
        # 写入目标文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"Successfully extracted {len(inputs)} inputs from {dataset_key} to {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # 设置命令行参数，python extract-alpaca-input.py --dataset=wj2015/psychology-10k-zh --key=train --output=dpo.txt
    parser = argparse.ArgumentParser(description="Extract input field from dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--key", type=str, required=True, help="Dataset key (e.g. train, test)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    
    # 调用提取函数
    extract_input_to_file(
        dataset_name=args.dataset,
        dataset_key=args.key,
        output_file=args.output
    )
