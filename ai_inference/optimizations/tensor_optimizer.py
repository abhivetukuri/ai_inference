import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass

@dataclass
class TensorStats:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    memory_usage: int
    is_contiguous: bool

class TensorOptimizer:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        self.stats_cache: Dict[str, TensorStats] = {}
    
    def analyze_tensor(self, tensor: torch.Tensor, name: str = "") -> TensorStats:
        return TensorStats(
            shape=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            memory_usage=tensor.element_size() * tensor.nelement(),
            is_contiguous=tensor.is_contiguous()
        )
    
    def optimize_memory_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_contiguous():
            return tensor.contiguous()
        return tensor
    
    def fuse_operations(self, ops: List[Tuple[str, torch.Tensor]]) -> torch.Tensor:
        result = None
        for op_name, tensor in ops:
            if op_name == "add":
                result = result + tensor if result is not None else tensor
            elif op_name == "mul":
                result = result * tensor if result is not None else tensor
        return result
    
    def cache_tensor(self, tensor: torch.Tensor, key: str):
        self.tensor_cache[key] = tensor
        self.stats_cache[key] = self.analyze_tensor(tensor, key)
    
    def get_cached_tensor(self, key: str) -> Optional[torch.Tensor]:
        return self.tensor_cache.get(key)

class MemoryOptimizer(TensorOptimizer):
    def __init__(self, device: Optional[str] = None, max_cache_size: int = 1024 * 1024 * 1024):
        super().__init__(device)
        self.max_cache_size = max_cache_size
        self.current_cache_size = 0
    
    def clear_unused_tensors(self):
        for key in list(self.tensor_cache.keys()):
            if not torch.is_tensor(self.tensor_cache[key]):
                del self.tensor_cache[key]
                del self.stats_cache[key]
    
    def optimize_memory_usage(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.float32:
            return tensor.half()
        return tensor

class ComputeOptimizer(TensorOptimizer):
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        self.supported_dtypes = {torch.float16, torch.float32, torch.bfloat16}
    
    def optimize_compute(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype not in self.supported_dtypes:
            return tensor.to(torch.float16)
        return tensor
    
    def batch_matmul(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            return None
        batch = torch.stack(tensors)
        return torch.matmul(batch, batch.transpose(-2, -1))

def create_optimizer(
    optimizer_type: str = "memory",
    device: Optional[str] = None,
    **kwargs
) -> TensorOptimizer:
    if optimizer_type == "memory":
        return MemoryOptimizer(device, **kwargs)
    elif optimizer_type == "compute":
        return ComputeOptimizer(device)
    return TensorOptimizer(device)

# Example usage:
if __name__ == "__main__":
    # Create sample tensors
    t1 = torch.randn(1000, 1000)
    t2 = torch.randn(1000, 1000)
    
    mem_opt = create_optimizer("memory", device="cpu")
    t1_opt = mem_opt.optimize_memory_usage(t1)
    print(f"Original dtype: {t1.dtype}, Optimized dtype: {t1_opt.dtype}")
    
    comp_opt = create_optimizer("compute", device="cpu")
    tensors = [t1, t2]
    result = comp_opt.batch_matmul(tensors)
    print(f"Batch matmul shape: {result.shape}") 