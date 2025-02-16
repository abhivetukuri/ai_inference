from ai_inference import ModelInference
from ai_inference.utils import (
    profile_memory,
    optimize_memory_usage,
    plot_memory_usage,
    log_memory_usage,
    estimate_memory_requirements,
    get_memory_usage,
    optimize_for_inference_speed
)
import time
import torch

def format_value(value):
    """Format a value for display, handling both numeric and string types."""
    if isinstance(value, (int, float)):
        return f"{value:.2f} GB"
    return str(value)

@profile_memory
def generate_with_profile(model, prompt, **kwargs):
    """Generate text with memory profiling."""
    return model.generate(prompt, **kwargs)

def run_benchmark():
    print("Starting AI Inference Benchmark...")
    
    # First, estimate memory requirements
    print("\nEstimating memory requirements...")
    memory_requirements = estimate_memory_requirements("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("Estimated memory requirements:")
    for key, value in memory_requirements.items():
        print(f"  {key}: {format_value(value)}")

    # Initialize model with optimizations
    print("\nInitializing model with optimizations...")
    model = ModelInference(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cpu",
        use_flash_attention=False
    )

    # Optimize memory usage
    print("\nOptimizing memory usage...")
    optimize_memory_usage(model.model)
    
    # Get initial memory usage
    initial_memory = get_memory_usage()
    print("\nInitial memory usage:")
    for key, value in initial_memory.items():
        print(f"  {key}: {format_value(value)}")

    # Test prompts for benchmarking
    test_prompts = [
        "Write a short poem about artificial intelligence.",
        "Explain the concept of quantum computing in simple terms.",
        "What are the key benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "What is the future of space exploration?"
    ]

    # Run inference with memory profiling
    print("\nRunning inference with memory profiling...")
    results = []
    memory_log = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nProcessing prompt {i}/{len(test_prompts)}")
        print(f"Prompt: {prompt[:50]}...")
        
        # Profile memory during inference
        start_time = time.time()
        output = generate_with_profile(
            model,
            prompt,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        end_time = time.time()
        
        # Log memory usage
        current_memory = get_memory_usage()
        memory_log.append(current_memory)
        
        # Store results
        results.append({
            "prompt_length": len(prompt),
            "output_length": len(output),
            "generation_time": end_time - start_time,
            "memory_usage": current_memory
        })
        
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        print("Memory usage:")
        for key, value in current_memory.items():
            print(f"  {key}: {format_value(value)}")

    # Plot memory usage
    print("\nGenerating memory usage plot...")
    plot_memory_usage(memory_log, "benchmark_memory.png")
    
    # Print summary statistics
    print("\nBenchmark Summary:")
    print("-" * 50)
    avg_time = sum(r["generation_time"] for r in results) / len(results)
    print(f"Average generation time: {avg_time:.2f} seconds")
    print(f"Total prompts processed: {len(results)}")
    print(f"Memory plot saved as: benchmark_memory.png")

if __name__ == "__main__":
    run_benchmark() 