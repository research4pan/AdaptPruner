import os
import gc
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    #load model
    try:
        tokenizer_kwargs = {}
        if "mobilellm" in args.base_model.lower():
            tokenizer_kwargs["use_fast"] = False
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tokenizer_kwargs)
        model_kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True, "torch_dtype": torch.float16}
        if "gemma" in args.base_model.lower():
            model_kwargs['attn_implementation'] = 'eager'
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            **model_kwargs
        )
    except:
        pruned_dict = torch.load(args.base_model, map_location='cpu', weights_only=False)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    
    model.to(args.device)
    logger.log(f"Original model: {args.base_model}")
    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("Original parameters: {}".format(before_pruning_parameters))
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = hf_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = hf_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = hf_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError
    
    layer_imp = None
    if args.adpative_prune:
        layer_imp_method = args.layer_imp_method.lower()
        assert layer_imp_method in ['cosine', 'euclidean', 'manhattan']
        if args.layer_imp_method == 'cosine':
            layer_imp = hf_pruner.cosine
            lower_is_better = True
        elif args.layer_imp_method == 'euclidean':
            layer_imp = hf_pruner.euclidean
            lower_is_better = True
        elif args.layer_imp_method == 'manhattan':
            layer_imp = hf_pruner.manhattan
            lower_is_better = True
        else:
            raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    
    if args.block_wise:
        kwargs = {
            "importance": imp,
            "layer_importance": layer_imp,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, 
            "consecutive_groups": {
                layer.self_attn.k_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "root_instances": [model.model.layers[i].self_attn.k_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        if 'llama' in args.base_model.lower():
            kwargs["customized_pruners"] = {LlamaRMSNorm: hf_pruner.hf_rmsnorm_pruner}
        elif 'qwen' in args.base_model.lower():
            kwargs["customized_pruners"] = {Qwen2RMSNorm: hf_pruner.hf_rmsnorm_pruner}
        elif 'gemma' in args.base_model.lower():
            kwargs["customized_pruners"] = {Gemma2RMSNorm: hf_pruner.hf_rmsnorm_pruner}
        else:
            customized_pruner = {}
            for module in model.modules():
                if any(x in module.__class__.__name__.lower() for x in ['rmsnorm', 'norm']):
                    customized_pruner[module.__class__] = hf_pruner.hf_rmsnorm_pruner
            kwargs["customized_pruners"] = customized_pruner
        
        logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()

        logger.log("Start Pruning")
        if before_pruning_parameters > args.target_param_num:
            for step in range(1, 1 + args.iterative_steps):
                example_prompts = get_examples(args.calibration_data_path, tokenizer, args.num_examples, seq_len = args.taylor_seq_len).to(args.device)
                if pruner_type == 'taylor':
                    logger.log("Start Backwarding in iterative steps = {}...".format(step))
                    total_loss = []
                    for mini_match in torch.split(example_prompts, args.batch_size):
                        loss = model(mini_match, labels=mini_match).loss
                        total_loss.append(loss)
                        loss.backward()
                    logger.log("Average Loss = {}".format(sum(total_loss) / len(total_loss)))

                if args.adpative_prune:
                    layer_imp_dict_by_index = pruner.adaptive_update_prune_distribution(example_prompts, lower_is_better, args.layer_prune_distribution_amplitude, args.batch_size)
                
                pruner.step()

                if pruner_type == 'taylor':
                    model.zero_grad()
                
                gc.collect()
                torch.cuda.empty_cache()
                    
                after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.log("After Iter {}/{}, #parameters: {}".format(step, args.iterative_steps, after_pruning_parameters))

                model = update_model_config_after_compression(model)

                if after_pruning_parameters < args.target_param_num:
                    break
                
                pruner.rebuild_DG(model)
        
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner
        gc.collect()
        torch.cuda.empty_cache()

    elif args.layer_wise:
        indices_to_remove = set(int(x) for x in args.prune_layer_idx.split(','))

        new_layers = torch.nn.ModuleList([
            layer for idx, layer in enumerate(model.model.layers)
            if idx not in indices_to_remove
        ])

        model.model.layers = new_layers
        model.config.num_hidden_layers = len(new_layers)
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, 'layer_idx'):
                layer.layer_idx = idx
            if hasattr(layer.self_attn, 'layer_idx'):
                layer.self_attn.layer_idx = idx

    else:
        raise NotImplementedError
    
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    logger.log(f"Compressed model's Configuration: {model.config}")

    output_dir = os.path.dirname(args.output_pth)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_model:
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, args.output_pth)
    logger.log(f"Pruned model saved at {args.output_pth}")


def update_model_config_after_compression(model):
    # modify inferece-related attributes
    num_attention_heads = []
    num_key_value_heads = []
    intermediate_size = []
    
    first_head_dim = model.model.layers[0].self_attn.head_dim
    
    for i, layer in enumerate(model.model.layers):
        assert layer.self_attn.head_dim == first_head_dim, \
            f"Layer {i} has inconsistent head_dim"
            
        num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        num_kv_heads = layer.self_attn.k_proj.weight.data.shape[0] // layer.self_attn.head_dim
        
        assert num_heads * layer.self_attn.head_dim == layer.self_attn.q_proj.weight.data.shape[0], \
            f"Layer {i}: Invalid num_heads calculation"
        assert num_kv_heads * layer.self_attn.head_dim == layer.self_attn.k_proj.weight.data.shape[0], \
            f"Layer {i}: Invalid num_kv_heads calculation"
            
        layer.self_attn.num_heads = num_heads
        layer.self_attn.num_key_value_heads = num_kv_heads
        layer.self_attn.hidden_size = num_heads * layer.self_attn.head_dim
        layer.mlp.intermediate_size = layer.mlp.gate_proj.out_features
        
        num_attention_heads.append(num_heads)
        num_key_value_heads.append(num_kv_heads)
        intermediate_size.append(layer.mlp.intermediate_size)
    
    model.config.head_dim = first_head_dim
    model.config.num_attention_heads = num_attention_heads
    model.config.num_key_value_heads = num_key_value_heads
    model.config.intermediate_size = intermediate_size
    model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='base model name')
    parser.add_argument('--save_log_name', type=str, default="llama_prune", help='the path for save the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--output_pth', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--block_wise', action='store_true', help='block wise pruning')
    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)
    
    parser.add_argument('--layer_wise', action='store_true', help='layer wise pruning')
    parser.add_argument('--prune_layer_idx', type=str, help='comma separated layer indices to prune, e.g. "13,18"', default="9,10,11,12,13,17,18,21")

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--target_param_num', type=int, default=1, help="target pruned param count")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--calibration_data_path', type=str, default="openhermes", help='data path for calibration dataset')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--taylor_seq_len', type=int, default=64, help='sequence length used for taylor pruning')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size used for taylor pruning')

    parser.add_argument('--adpative_prune', action='store_true', help='whether prune adaptively among different layers')
    parser.add_argument('--layer_imp_method', type=str, default='cosine', help='the eval method to determine which layer is important')
    parser.add_argument('--layer_prune_distribution_amplitude', type=float, default=0.03, help='the amplitude of prune sparsity among different layers')

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
