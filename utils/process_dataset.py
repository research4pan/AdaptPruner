import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import argparse
import math
import shutil
import multiprocessing
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.post_training.load_dataset import get_train_val_data

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

    num_proc = min(64, int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count())) - 2)
    logger.log(f"Using num_proc: {num_proc}")

    logger.log(f"Processing dataset from {args.data_path}")
    combined_train, _ = get_train_val_data(
        data_path=args.data_path,
        tokenizer=tokenizer,
        cutoff_len=args.cutoff_len,
        val_set_size=0,
        num_proc=num_proc,
        fraction=args.fraction,
    )

    if args.repeat_dataset > 1:
        logger.log(f"Repeating dataset {args.repeat_dataset} times")
        repeated_datasets = [combined_train for _ in range(args.repeat_dataset)]
        combined_train = concatenate_datasets(repeated_datasets)
        logger.log(f"Dataset size after repeating: {len(combined_train)}")
        
        logger.log("Shuffling dataset...")
        combined_train = combined_train.shuffle(seed=42)
        logger.log("Dataset shuffled")
    
    total_size = len(combined_train)
    
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    cumulative_sizes=[]
    for i in range(args.iterative_prune_train_step):
        if args.seperate_strategy == 'linear':
            ratio = 2 * (i + 1) / (args.iterative_prune_train_step * (args.iterative_prune_train_step + 1))
            current_part_size = int(total_size * ratio)
        elif args.seperate_strategy == 'constant':
            current_part_size = total_size // args.iterative_prune_train_step
        elif args.seperate_strategy == 'cosine':
            theta = (i+1) * math.pi / (2 * (args.iterative_prune_train_step))
            ratio = (1 - math.cos(theta))
            current_part_size = int(total_size * ratio - (int(total_size * (1 - math.cos(i * math.pi / (2 * (args.iterative_prune_train_step)))))))
        else:
            raise NotImplementedError
        
        if i == 0:
            start_idx = 0
        else:
            start_idx = end_idx
        
        end_idx = start_idx + current_part_size
        if i == args.iterative_prune_train_step - 1:
            end_idx = total_size

        cumulative_sizes.append(end_idx-start_idx)
        
        logger.log(f"Processing part {i+1}/{args.iterative_prune_train_step}")
        part_dataset = combined_train.select(range(start_idx, end_idx))
        
        save_path = os.path.join(args.output_dir, f'tokenized_dataset_stage_{i+1}_of_{args.iterative_prune_train_step}')
        part_dataset.save_to_disk(save_path, num_proc=num_proc)
        
        logger.log(f"Saved part {i+1} with {end_idx - start_idx} samples to {save_path}")
    
    logger.log("\nFinal size distribution:")
    logger.log(f"Strategy: {args.seperate_strategy}")
    logger.log(f"Total size: {total_size}")
    for i, size in enumerate(cumulative_sizes):
        logger.log(f"Part {i+1}: {size} samples ({(size/total_size*100):.2f}%)")
    logger.log(f"Sum of all parts: {sum(cumulative_sizes)}")

    with open(f"{args.return_pth}", "w") as f:
        f.write(str(int(total_size)))
    return total_size

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
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='fraction of datasets to use (default: 1.0)')
    
    args = parser.parse_args()
    main(args)
