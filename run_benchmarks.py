"""
Script to run benchmarks and update README with results.
"""

import os
import sys
from pathlib import Path
from ai_inference.benchmark import run_comprehensive_benchmark, generate_markdown_table
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from ai_inference.model import ModelInference
import psutil

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def get_cpu_memory_usage():
    """Get current CPU memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def run_benchmark(model_id, quantization="4bit", use_flash_attention=True):
    """Run a single benchmark with given configuration."""
    print(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model with {quantization} quantization and flash attention: {use_flash_attention}")
    model = ModelInference(
        model_id=model_id,
        quantization=quantization,
        use_flash_attention=use_flash_attention
    )
    
    # Prepare input
    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Warm up
    print("Warming up...")
    for _ in range(3):
        model.generate(inputs["input_ids"], max_length=50)
    
    # Benchmark
    print("Running benchmark...")
    num_runs = 5
    total_time = 0
    
    for i in range(num_runs):
        start_time = time.time()
        outputs = model.generate(inputs["input_ids"], max_length=50)
        end_time = time.time()
        total_time += end_time - start_time
        
        if i == 0:  # Print first run's output
            print(f"Sample output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    avg_time = total_time / num_runs
    gpu_memory = get_gpu_memory_usage()
    cpu_memory = get_cpu_memory_usage()
    
    return {
        "model_id": model_id,
        "quantization": quantization,
        "use_flash_attention": use_flash_attention,
        "avg_time": avg_time,
        "gpu_memory": gpu_memory,
        "cpu_memory": cpu_memory
    }

def update_readme(results):
    """Update README with benchmark results."""
    readme_path = "README.md"
    with open(readme_path, "r") as f:
        readme = f.read()
    
    # Find the benchmark results section
    benchmark_section = "## Benchmark Results\n\n"
    if benchmark_section not in readme:
        readme += f"\n{benchmark_section}"
    
    # Generate results table
    table = "| Model | Quantization | Flash Attention | Avg Time (s) | GPU Memory (GB) | CPU Memory (GB) |\n"
    table += "|-------|--------------|-----------------|--------------|----------------|----------------|\n"
    
    for result in results:
        table += f"| {result['model_id']} | {result['quantization']} | {result['use_flash_attention']} | {result['avg_time']:.3f} | {result['gpu_memory']:.2f} | {result['cpu_memory']:.2f} |\n"
    
    # Update the benchmark section
    benchmark_start = readme.find(benchmark_section) + len(benchmark_section)
    benchmark_end = readme.find("\n##", benchmark_start)
    if benchmark_end == -1:
        benchmark_end = len(readme)
    
    new_readme = readme[:benchmark_start] + table + readme[benchmark_end:]
    
    with open(readme_path, "w") as f:
        f.write(new_readme)

def main():
    """Run comprehensive benchmarks."""
    print("Running comprehensive benchmarks...")
    
    # List of models to benchmark
    model_ids = [
        "bigscience/bloom-560m",  # Small BLOOM model for testing
    ]
    
    # Test configurations
    configs = [
        {"quantization": "4bit", "use_flash_attention": True},
        {"quantization": "4bit", "use_flash_attention": False},
        {"quantization": "8bit", "use_flash_attention": True},
        {"quantization": "8bit", "use_flash_attention": False},
    ]
    
    # Run benchmarks for each model and configuration
    results = []
    for model_id in model_ids:
        print(f"\nBenchmarking {model_id}")
        for config in configs:
            try:
                result = run_benchmark(model_id, **config)
                results.append(result)
            except Exception as e:
                print(f"Failed to run benchmark for {model_id} with config {config}: {str(e)}")
    
    # Update README with results
    update_readme(results)
    print("\nBenchmarks completed and README updated!")

if __name__ == "__main__":
    main() 