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

    num_proc = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count())) - 2
    logger.log(f"Using num_proc: {num_proc}")

    logger.log(f"Processing dataset from {args.data_path}")
    combined_train, _ = get_train_val_data(
        data_path=args.data_path,
        tokenizer=tokenizer,
        cutoff_len=args.cutoff_len,
        val_set_size=0,
        num_proc=num_proc
    )

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
