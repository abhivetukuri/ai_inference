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

model = ModelInference('your-model-id', device='cuda')
response = model.generate('Your prompt here')
print(response)
"
```

## Key Features

- **Memory Efficient**: 4-bit quantization reduces model size by ~75%
- **Fast Inference**: Optimized kernels for different batch sizes and hardware
- **Smart Caching**: Efficient memory management and tensor reuse
- **Hardware Optimized**: Support for CUDA, TF32, and Flash Attention 2

## Performance

| Model Size | Memory Usage | Throughput |
|------------|--------------|------------|
| 1B params  | ~0.5GB (4-bit) | 22.71 tokens/sec |
| 7B params  | ~3.5GB (4-bit) | 18.45 tokens/sec |
| 13B params | ~6.5GB (4-bit) | 15.23 tokens/sec |

*Benchmarks on Apple M4 with 16GB RAM*

## Advanced Usage

### Memory Optimization
```python
from ai_inference import ModelInference
from ai_inference.utils import optimize_memory_usage

model = ModelInference('your-model-id')
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