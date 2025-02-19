# AI Inference

A high-performance PyTorch inference engine optimized for large language models. Built with memory efficiency and speed in mind.

## Quick Start

```bash
# Install
git clone https://github.com/yourusername/ai_inference.git
cd ai_inference
poetry install

# Run inference
python -c "
from ai_inference import ModelInference

model = ModelInference('your-model-id', device='cuda', quantization='4bit')
response = model.generate('Your prompt here')
print(response)
"
```

## Key Features

- **Memory Efficient**: Multiple quantization options (4-bit, 8-bit, FP16)
- **Fast Inference**: Optimized kernels for different batch sizes and hardware
- **Smart Caching**: Efficient memory management and tensor reuse
- **Hardware Optimized**: Support for CUDA, TF32, and Flash Attention 2

## Performance

| Model Size | Quantization | Memory Usage | Throughput |
|------------|--------------|--------------|------------|
| 1B params  | 4-bit       | ~0.5GB      | 22.71 tokens/sec |
| 1B params  | 8-bit       | ~1.0GB      | 24.15 tokens/sec |
| 7B params  | 4-bit       | ~3.5GB      | 18.45 tokens/sec |
| 7B params  | 8-bit       | ~7.0GB      | 19.82 tokens/sec |
| 13B params | 4-bit       | ~6.5GB      | 15.23 tokens/sec |
| 13B params | 8-bit       | ~13.0GB     | 16.45 tokens/sec |

*Benchmarks on Apple M4 with 16GB RAM*

## Advanced Usage

### Memory Optimization
```python
from ai_inference import ModelInference
from ai_inference.utils import optimize_memory_usage

# Choose quantization based on your needs
model = ModelInference('your-model-id', quantization='8bit')  # Better accuracy, more memory
model = ModelInference('your-model-id', quantization='4bit')  # Less memory, slightly lower accuracy
model = ModelInference('your-model-id', quantization='fp16')  # Full precision, maximum memory

optimize_memory_usage(model.model)
```

### Benchmarking
```bash
# Run comprehensive benchmarks
python benchmark.py

# Quick memory estimation
python main.py --estimate_memory --model your-model-id
```

### Command Line Options
```bash
# Speed optimization
python main.py --max_speed

# Memory optimization
python main.py --optimize_memory

# Custom batch size
python main.py --batch_size 4
```

## Technical Details

### Memory Requirements
- FP16: ~2GB per 1B parameters
- 8-bit: ~1GB per 1B parameters
- 4-bit: ~0.5GB per 1B parameters
- Activation memory: ~0.34GB
- KV cache: ~0.34GB

### Optimizations
- Vectorized matrix operations
- Kernel fusion
- Contiguous memory layouts
- Strategic tensor clearing
- Hardware-specific implementations

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (optional)
- 8GB+ RAM

## License

MIT 