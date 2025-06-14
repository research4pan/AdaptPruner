import os
import gc
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.eval import print_memory_usage, print_gpu_memory, print_model_architecture_and_parameters
from LLMPruner.evaluator.benchmark_eval import eval_tasks_performance
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts
from LLMPruner.post_training.load_dataset import get_train_val_data

import warnings
warnings.filterwarnings("ignore")

def set_random_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )
    
    best_optimal_score=float('-inf')
    optimal_model_path=None
    optimal_model=None
    optimal_tokenizer=None
    model_paths = [path.strip() for path in args.model_paths.split(',')]
    if not model_paths:
        raise ValueError("No model paths provided")
    logger.log(model_paths)
    for model_path in model_paths:
        logger.log(f"Evaluating model: {model_path}")
        # Load Pruned Model
        try:
            tokenizer_kwargs = {}
            if "mobilellm" in model_path.lower():
                tokenizer_kwargs["use_fast"] = False
            tokenizer_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )
        except:
            pruned_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
            model = model.to(torch.bfloat16)
        
        tokenizer.pad_token_id = 0 
        model.config.pad_token_id = 0 
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.generation_config.pad_token_id = 0 
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        tokenizer.padding_side = "left"

        model = model.to(args.eval_device)
        model.eval()

        '''logger.log("\n==================Generation Results after Pruning================\n")
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_new_tokens=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)'''

        tasks=["arc_easy", "arc_challenge", "hellaswag", "openbookqa", "piqa", "winogrande", "social_iqa"]
        result_table, avg_score = eval_tasks_performance(model, tokenizer, tasks=tasks, num_fewshot=0)
        logger.log("0-shot tasks performance before pruning: \n{}".format(result_table))
        logger.log("Average score across all tasks: {:.4f}".format(avg_score))

        tasks=["wikitext"]
        result_table, _ = eval_tasks_performance(model, tokenizer, tasks=tasks, num_fewshot=0)
        logger.log("Perplexity before pruning: \n{}".format(result_table))



        for param in model.parameters():
            param.requires_grad_(True)

        before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log("Model parameters: {}".format(before_pruning_parameters))
        logger.log("Model architecture: {}".format(model.config))

        if avg_score > best_optimal_score:
            if optimal_model is not None:
                del optimal_model
                del optimal_tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                
            best_optimal_score = avg_score
            optimal_model = copy.deepcopy(model).to('cpu')
            optimal_tokenizer = copy.deepcopy(tokenizer)
            optimal_model_path=model_path
            logger.log(f"Update optimal model path: {optimal_model_path}")
            
        
        model = model.cpu()
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    if len(model_paths) > 1:
        optimal_model.save_pretrained(args.output_dir)
        optimal_tokenizer.save_pretrained(args.output_dir)
        logger.log(f"Optimal model path: {optimal_model_path}")
        logger.log(f"Optimal model saved at: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--model_paths', type=str, default="meta-llama/Llama-2-7b-hf", help='base model name')
    parser.add_argument('--save_log_name', type=str, default="llama_prune", help='the path for save the log.')
    parser.add_argument('--output_dir', type=str, default="llama_prune", help='the path for saving the optimal model')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
