import time
import torch
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import json
import os
from pathlib import Path

from .model import ModelInference
from .utils import get_memory_usage, estimate_memory_requirements

@dataclass
class BenchmarkResult:
    model_size: str
    quantization: str
    use_flash_attention: bool
    batch_size: int
    memory_usage_gb: float
    throughput_tokens_per_sec: float
    latency_ms: float
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    cpu_percent: float
    ram_percent: float

def run_benchmark(
    model_id: str,
    quantization: str = "4bit",
    use_flash_attention: bool = True,
    batch_size: Optional[int] = None,
    num_runs: int = 5,
    seq_length: int = 512,
    num_tokens: int = 100,
    warmup_runs: int = 2,
) -> BenchmarkResult:
    model = ModelInference(
        model_id=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_flash_attention=use_flash_attention,
        quantization=quantization,
        dynamic_batch_size=batch_size is None
    )
    
    prompts = ["The quick brown fox jumps over the lazy dog."] * (batch_size or model.optimal_batch_size)
    
    for _ in range(warmup_runs):
        model.generate(prompts[0], max_length=num_tokens)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    latencies = []
    token_counts = []
    
    for _ in tqdm(range(num_runs), desc=f"Running benchmark with {quantization} quantization"):
        start_time = time.time()
        outputs = model.generate(prompts, max_length=num_tokens)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        num_tokens_generated = sum(len(output.split()) for output in outputs)
        
        latencies.append(latency)
        token_counts.append(num_tokens_generated)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    avg_latency = np.mean(latencies)
    avg_tokens = np.mean(token_counts)
    throughput = (avg_tokens / (avg_latency / 1000))
    
    memory_stats = get_memory_usage()
    
    model_size = estimate_memory_requirements(model_id)
    
    return BenchmarkResult(
        model_size=model_id,
        quantization=quantization,
        use_flash_attention=use_flash_attention,
        batch_size=batch_size or model.optimal_batch_size,
        memory_usage_gb=model_size[f"total_{quantization}_gb"],
        throughput_tokens_per_sec=throughput,
        latency_ms=avg_latency,
        gpu_memory_allocated_gb=memory_stats.get("cuda_allocated_gb", 0),
        gpu_memory_reserved_gb=memory_stats.get("cuda_reserved_gb", 0),
        cpu_percent=memory_stats.get("cpu_percent", 0),
        ram_percent=memory_stats.get("ram_percent", 0)
    )

def run_comprehensive_benchmark(
    model_ids: List[str],
    output_file: str = "benchmark_results.json",
) -> List[BenchmarkResult]:
    results = []
    configs = [
        {"quantization": "4bit", "use_flash_attention": True},
        {"quantization": "4bit", "use_flash_attention": False},
        {"quantization": "8bit", "use_flash_attention": True},
        {"quantization": "8bit", "use_flash_attention": False},
    ]
    
    for model_id in model_ids:
        print(f"\nBenchmarking {model_id}")
        for config in configs:
            try:
                result = run_benchmark(
                    model_id=model_id,
                    **config
                )
                results.append(result)
                print(f"Completed {config['quantization']} with Flash Attention {config['use_flash_attention']}")
            except Exception as e:
                print(f"Failed to run benchmark for {model_id} with config {config}: {e}")
    
    with open(output_file, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    
    return results

def generate_markdown_table(results: List[BenchmarkResult]) -> str:
    model_groups = {}
    for result in results:
        if result.model_size not in model_groups:
            model_groups[result.model_size] = []
        model_groups[result.model_size].append(result)
    
    table = "| Model Size | Quantization | Flash Attention | Memory Usage | Throughput |\n"
    table += "|------------|--------------|-----------------|--------------|------------|\n"
    
    for model_size, model_results in model_groups.items():
        for result in model_results:
            table += f"| {model_size} | {result.quantization} | {'Yes' if result.use_flash_attention else 'No'} | "
            table += f"~{result.memory_usage_gb:.1f}GB | {result.throughput_tokens_per_sec:.2f} tokens/sec |\n"
    
    return table

if __name__ == "__main__":
    model_ids = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf"
    ]
    
    results = run_comprehensive_benchmark(model_ids)
    markdown_table = generate_markdown_table(results)
    print("\nBenchmark Results:")
    print(markdown_table)