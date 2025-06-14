import torch
import psutil
import os

def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")


def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_memory = memory_info.rss / (1024 ** 3)
    print(f"Current memory usage: {rss_memory:.2f} GB")


def print_model_architecture_and_parameters(model):
        total_params = 0
        total_weight_memory_usage = 0
        total_activation_memory_usage = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_weight_memory_usage = param.numel() * param.element_size()
                total_params += param.numel()
                total_weight_memory_usage += layer_weight_memory_usage
        print(f"Total number of trainable parameters: {total_params}")
        print(f"Total weight memory usage (GB): {total_weight_memory_usage / (1024 ** 3):.3f}")
        print(f"Average weight memory usage per parameter (bit): {total_weight_memory_usage * 8 / total_params}")
        print(f"Total activation memory usage (GB): {total_activation_memory_usage / (1024 ** 3):.3f}")
        print(f"Average activation memory usage per parameter (bit): {total_activation_memory_usage * 8 / total_params}")
        total_memory_usage = total_weight_memory_usage + total_activation_memory_usage
        print(f"Total memory usage (GB): {total_memory_usage / (1024 ** 3):.3f}")
        print(f"Average memory usage per parameter (bit): {total_memory_usage * 8 / total_params}")