import gc
import torch
import inspect
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import time

class GPUMemoryTracker:
    def __init__(self):
        self.tensor_refs = defaultdict(list)
    
    def track_tensors(self):
        self.tensor_refs.clear()
        
        # 特别跟踪模型参数
        for name, module in list(inspect.currentframe().f_back.f_locals.items()):
            if isinstance(module, torch.nn.Module):
                params_size = sum(p.numel() * p.element_size() 
                                for p in module.parameters()) / 1024 / 1024
                print(f"Model {name} parameters size: {params_size:.2f} MB")
        
        # 原来的张量跟踪逻辑
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:  # 只跟踪 CUDA 张量
                    frame = inspect.currentframe()
                    while frame:
                        if frame.f_code.co_name != 'track_tensors':
                            break
                        frame = frame.f_back
                    
                    name = frame.f_code.co_name if frame else "unknown"
                    self.tensor_refs[name].append(obj)
            except:
                pass
        
        memory_usage = {}
        for name, tensors in self.tensor_refs.items():
            total_memory = sum(t.element_size() * t.nelement() for t in tensors)
            memory_usage[name] = total_memory / 1024 / 1024
        
        return memory_usage
    
    def print_memory_usage(self):
        memory_usage = self.track_tensors()
        
        print("\n=== GPU Memory Usage ===")
        print("\n-- Tensors by Context --")
        for name, memory in memory_usage.items():
            print(f"{name}: {memory:.2f} MB")
        
        if torch.cuda.is_available():
            print("\n-- Overall GPU Memory --")
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Current Allocated: {allocated:.2f} MB")
            print(f"Max Allocated: {max_allocated:.2f} MB")
            print(f"Reserved: {reserved:.2f} MB")

@dataclass
class MemorySnapshot:
    allocated: float  # GB
    reserved: float  # GB
    active_tensors: Dict[str, float]  # name -> size in GB
    timestamp: float

class DetailedMemoryProfiler:
    def __init__(self):
        self.snapshots = []
        self.reset_cuda_stats()
    
    def reset_cuda_stats(self):
        """Reset CUDA memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def _get_tensor_info(self, obj) -> Optional[Dict[str, Any]]:
        """Get information about a tensor if it's a CUDA tensor"""
        if hasattr(obj, 'device') and str(obj.device).startswith('cuda'):
            return {
                'size': obj.element_size() * obj.nelement() / 1024**3,  # GB
                'shape': list(obj.shape),
                'dtype': str(obj.dtype),
                'device': str(obj.device)
            }
        return None

    def take_snapshot(self, name: str = ""):
        """Take a snapshot of current memory state"""
        if not torch.cuda.is_available():
            return
        
        # Get basic CUDA memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        
        # Track all CUDA tensors
        tensor_sizes = defaultdict(float)
        for obj in gc.get_objects():
            try:
                info = self._get_tensor_info(obj)
                if info:
                    key = f"{name}_{tuple(info['shape'])}_{info['dtype']}"
                    tensor_sizes[key] += info['size']
            except:
                continue
        
        # Create and store snapshot
        snapshot = MemorySnapshot(
            allocated=allocated,
            reserved=reserved,
            active_tensors=dict(tensor_sizes),
            timestamp=time.time()
        )
        self.snapshots.append(snapshot)
        
        # Print current state
        print(f"\n=== Memory Snapshot: {name} ===")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print("\nLargest tensors:")
        sorted_tensors = sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True)
        for tensor_name, size in sorted_tensors[:10]:  # Top 10 largest tensors
            if size > 0.01:  # Only show tensors larger than 10MB
                print(f"{tensor_name}: {size:.2f} GB")
    
    def print_model_size(self, model: torch.nn.Module, name: str = "model"):
        """Print detailed model size information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        
        print(f"\n=== {name} Size Analysis ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Parameter memory: {param_size:.2f} GB")
        
        # Size by layer type
        layer_sizes = defaultdict(lambda: {"params": 0, "memory": 0})
        for name, module in model.named_modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            memory = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False)) / 1024**3
            if params > 0:
                layer_type = module.__class__.__name__
                layer_sizes[layer_type]["params"] += params
                layer_sizes[layer_type]["memory"] += memory
        
        print("\nSize by layer type:")
        for layer_type, sizes in layer_sizes.items():
            if sizes["memory"] > 0:
                print(f"{layer_type}: {sizes['params']:,} params, {sizes['memory']:.2f} GB")
    
    def print_optimizer_size(self, optimizer):
        """Print optimizer state size"""
        total_state_size = 0
        state_sizes = defaultdict(float)
        
        for group in optimizer.state_dict()['state'].values():
            for key, value in group.items():
                if isinstance(value, torch.Tensor):
                    size = value.numel() * value.element_size() / 1024**3
                    total_state_size += size
                    state_sizes[key] += size
        
        print("\n=== Optimizer Memory Analysis ===")
        print(f"Total optimizer state size: {total_state_size:.2f} GB")
        print("\nSize by state type:")
        for state_type, size in state_sizes.items():
            print(f"{state_type}: {size:.2f} GB")
    
    def compare_snapshots(self, snapshot1_idx: int, snapshot2_idx: int):
        """Compare two memory snapshots to see what changed"""
        if len(self.snapshots) <= max(snapshot1_idx, snapshot2_idx):
            return
        
        s1 = self.snapshots[snapshot1_idx]
        s2 = self.snapshots[snapshot2_idx]
        
        print(f"\n=== Memory Change Analysis ===")
        print(f"Allocated: {s1.allocated:.2f} GB -> {s2.allocated:.2f} GB ({s2.allocated - s1.allocated:.2f} GB)")
        print(f"Reserved:  {s1.reserved:.2f} GB -> {s2.reserved:.2f} GB ({s2.reserved - s1.reserved:.2f} GB)")
        
        # Compare tensor sizes
        all_keys = set(s1.active_tensors.keys()) | set(s2.active_tensors.keys())
        changes = {}
        for key in all_keys:
            old_size = s1.active_tensors.get(key, 0)
            new_size = s2.active_tensors.get(key, 0)
            if abs(new_size - old_size) > 0.01:  # Only show significant changes
                changes[key] = new_size - old_size
        
        print("\nLargest memory changes:")
        for key, change in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"{key}: {change:.2f} GB")
