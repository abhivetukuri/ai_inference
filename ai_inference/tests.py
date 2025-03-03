"""
Tests for AI inference optimizations.
"""

import unittest
import torch
import numpy as np
from typing import List, Dict
import time
import os

from .model import ModelInference
from .attention import OptimizedAttention
from .utils import get_memory_usage, calculate_optimal_batch_size

class TestModelInference(unittest.TestCase):
    """Test cases for ModelInference class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    def test_quantization(self):
        """Test different quantization options."""
        quantizations = ["4bit", "8bit", "fp16"]
        for q in quantizations:
            model = ModelInference(
                self.model_id,
                device=self.device,
                quantization=q
            )
            self.assertEqual(model.quantization, q)
            
            # Test generation
            output = model.generate("Hello, world!")
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 0)
    
    def test_flash_attention(self):
        """Test Flash Attention 2 integration."""
        # Test with Flash Attention
        model_with_flash = ModelInference(
            self.model_id,
            device=self.device,
            use_flash_attention=True
        )
        
        # Test without Flash Attention
        model_without_flash = ModelInference(
            self.model_id,
            device=self.device,
            use_flash_attention=False
        )
        
        # Compare outputs
        prompt = "The quick brown fox jumps over the lazy dog."
        output_with_flash = model_with_flash.generate(prompt)
        output_without_flash = model_without_flash.generate(prompt)
        
        # Outputs should be similar but not identical due to sampling
        self.assertGreater(len(output_with_flash), 0)
        self.assertGreater(len(output_without_flash), 0)
    
    def test_dynamic_batching(self):
        """Test dynamic batch sizing."""
        model = ModelInference(
            self.model_id,
            device=self.device,
            dynamic_batch_size=True
        )
        
        # Test optimal batch size calculation
        optimal_batch_size = calculate_optimal_batch_size(
            model.model,
            safety_margin=0.8
        )
        self.assertGreater(optimal_batch_size, 0)
        
        # Test batch generation
        prompts = ["Hello", "World", "Test"] * 2
        outputs = model.batch_generate(prompts)
        self.assertEqual(len(outputs), len(prompts))
    
    def test_memory_optimization(self):
        """Test memory optimization features."""
        model = ModelInference(
            self.model_id,
            device=self.device,
            quantization="4bit",
            use_flash_attention=True
        )
        
        # Test memory usage tracking
        memory_stats = model.get_memory_usage()
        self.assertIsInstance(memory_stats, dict)
        
        if self.device == "cuda":
            self.assertIn("cuda_allocated_gb", memory_stats)
            self.assertIn("cuda_reserved_gb", memory_stats)
        
        # Test memory snapshot
        snapshot_file = "test_memory_snapshot.txt"
        model.save_memory_snapshot(snapshot_file)
        self.assertTrue(os.path.exists(snapshot_file))
        os.remove(snapshot_file)
    
    def test_attention_optimization(self):
        """Test attention layer optimization."""
        model = ModelInference(
            self.model_id,
            device=self.device,
            use_flash_attention=True
        )
        
        # Verify attention layers were replaced
        attention_layers = [
            module for module in model.model.modules()
            if isinstance(module, OptimizedAttention)
        ]
        self.assertGreater(len(attention_layers), 0)
        
        # Test attention computation
        test_input = torch.randn(1, 10, model.model.config.hidden_size).to(self.device)
        for layer in attention_layers:
            output = layer(test_input)
            self.assertEqual(output.shape, test_input.shape)
    
    def test_performance(self):
        """Test performance improvements."""
        # Test with optimizations
        model_optimized = ModelInference(
            self.model_id,
            device=self.device,
            quantization="4bit",
            use_flash_attention=True
        )
        
        # Test without optimizations
        model_unoptimized = ModelInference(
            self.model_id,
            device=self.device,
            quantization="fp16",
            use_flash_attention=False
        )
        
        # Measure generation time
        prompt = "The quick brown fox jumps over the lazy dog."
        num_runs = 5
        
        # Test optimized model
        optimized_times = []
        for _ in range(num_runs):
            start_time = time.time()
            model_optimized.generate(prompt)
            optimized_times.append(time.time() - start_time)
        
        # Test unoptimized model
        unoptimized_times = []
        for _ in range(num_runs):
            start_time = time.time()
            model_unoptimized.generate(prompt)
            unoptimized_times.append(time.time() - start_time)
        
        # Compare average times
        avg_optimized = np.mean(optimized_times)
        avg_unoptimized = np.mean(unoptimized_times)
        
        # Optimized version should be faster
        self.assertLess(avg_optimized, avg_unoptimized * 1.5)  # Allow 50% margin
    
    def test_error_handling(self):
        """Test error handling and fallbacks."""
        # Test invalid quantization
        with self.assertRaises(ValueError):
            ModelInference(
                self.model_id,
                device=self.device,
                quantization="invalid"
            )
        
        # Test memory overflow handling
        model = ModelInference(
            self.model_id,
            device=self.device,
            quantization="4bit"
        )
        
        # Generate with very long sequence
        long_prompt = "Hello " * 1000
        output = model.generate(long_prompt, max_length=100)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

if __name__ == "__main__":
    unittest.main() 