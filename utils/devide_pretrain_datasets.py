import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import argparse
import torch
from datasets import Dataset
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np

from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.post_training.load_dataset import get_train_val_data

def process_chunk(args):
    item, chunk_size = args
    current_length = len(item['input_ids'])
    chunks = []
    
    for i in range(0, current_length, chunk_size):
        end_idx = min(i + chunk_size, current_length)
        chunks.append({
            'input_ids': item['input_ids'][i:end_idx],
            'attention_mask': item['attention_mask'][i:end_idx],
            'labels': item['labels'][i:end_idx]
        })
    
    return chunks

def chunk_and_save_dataset_parallel(dataset, chunk_size=1024, save_path='/work/hdd/bdjz/boyaow2/dataset/combined_pretrain_11B/devided', 
                                  batch_size=1024, num_proc=16):
    if num_proc is None:
        num_proc = max(1, multiprocessing.cpu_count() - 2)
    
    print(f"使用{num_proc}个进程进行并行处理")
    
    # 准备数据
    all_chunks = []
    total_items = len(dataset)
    
    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        # 批量处理数据
        for batch_start in tqdm(range(0, total_items, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = dataset[batch_start:batch_end]
            
            # 并行处理当前批次
            chunk_futures = list(executor.map(
                process_chunk, 
                [(item, chunk_size) for item in batch_items]
            ))
            
            # 收集结果
            for chunks in chunk_futures:
                all_chunks.extend(chunks)
    
    # 转换为Dataset格式
    chunked_input_ids = [chunk['input_ids'] for chunk in all_chunks]
    chunked_attention_mask = [chunk['attention_mask'] for chunk in all_chunks]
    chunked_labels = [chunk['labels'] for chunk in all_chunks]
    
    # 创建新的数据集
    new_dataset = Dataset.from_dict({
        'input_ids': chunked_input_ids,
        'attention_mask': chunked_attention_mask,
        'labels': chunked_labels
    })
    
    # 打乱数据集
    new_dataset = new_dataset.shuffle(seed=42)
    
    # 保存数据集
    print(f"保存数据集到{save_path}")
    new_dataset.save_to_disk(save_path,num_proc=num_proc)
    
    # 打印统计信息
    print(f"已将数据集切分为{len(new_dataset)}个chunks并保存到{save_path}")
    print(f"每个chunk最大长度为{chunk_size} tokens")
    total_tokens = sum(len(x) for x in new_dataset['input_ids'])
    print(f"总共包含{total_tokens:,}个tokens")
    
    return new_dataset

def main(args):
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    logger.log(f"Loading tokenizer from {args.model}")
    tokenizer_kwargs = {}
    if "mobilellm" in args.model.lower():
        tokenizer_kwargs["use_fast"] = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    
    data_path="/work/hdd/bdjz/boyaow2/dataset/books_dataset"
    train_data1, _ = get_train_val_data(data_path,tokenizer,409600000,0)

    data_path="DKYoon/SlimPajama-6B"
    train_data2, _ = get_train_val_data(data_path,tokenizer,409600000,0)

    train_data = concatenate_datasets([train_data1, train_data2])

    chunked_dataset = chunk_and_save_dataset_parallel(train_data, chunk_size=1024)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and separate datasets')

    parser.add_argument('--model', type=str, required=True,
                        help='Model name or path to use for tokenization')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Data path to process and save')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed datasets')
    parser.add_argument('--return_pth', type=str, default='return_value.out',
                        help='path to save return value')
    parser.add_argument('--save_log_name', type=str, required=True,
                        help='Directory to save logging infor')
    parser.add_argument('--cutoff_len', type=int, default=1024,
                        help='Cutoff length for dataset (default: 1024)')
    parser.add_argument('--seperate_strategy', type=str, default='linear',
                        help='strategy to seperate the dataset across stages (default: linear)')
    parser.add_argument('--iterative_prune_train_step', type=int, default=3,
                        help='iterative_prune_train_step (default: 3)')
    parser.add_argument('--repeat_dataset', type=int, default=1,
                        help='Number of times to repeat the dataset (default: 1)')

    args = parser.parse_args()
    main(args)
