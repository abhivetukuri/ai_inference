"""
Model wrapper for efficient inference with AI models using PyTorch.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.utils import logging
from typing import List, Optional, Union, Dict, Any, Tuple, Literal

from .pytorch_kernels import Linear4Bit, Linear8Bit
from .utils import get_memory_usage, calculate_optimal_batch_size
from .attention import OptimizedAttention


class ModelInference:
    """
    Memory-efficient inference for AI models using PyTorch.
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        use_flash_attention: bool = True,
        quantization: Literal["4bit", "8bit", "fp16"] = "4bit",
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        offload_folder: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        cache_dir: Optional[str] = None,
        dynamic_batch_size: bool = True,
        safety_margin: float = 0.8,
    ):
        """
        Initialize the model with PyTorch for memory-efficient inference.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to load the model on ('cuda', 'cpu')
            use_flash_attention: Whether to use flash attention for faster inference
            quantization: Quantization type to use ('4bit', '8bit', or 'fp16')
            max_memory: Maximum memory to use for each GPU
            offload_folder: Folder to offload weights to
            low_cpu_mem_usage: Whether to use low CPU memory usage when loading
            cache_dir: Directory to cache models
            dynamic_batch_size: Whether to use dynamic batch sizing
            safety_margin: Safety margin for dynamic batch sizing
        """
        self.model_id = model_id
        self.device = device
        self.quantization = quantization
        self.dynamic_batch_size = dynamic_batch_size
        self.safety_margin = safety_margin
        self.use_flash_attention = use_flash_attention
        
        # Set up logging
        logging.set_verbosity_info()
        self.logger = logging.get_logger("transformers")
        
        # Load tokenizer
        self.logger.info(f"Loading tokenizer from {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        
        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.logger.info("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Handle case where eos_token_id is a list
            if isinstance(self.tokenizer.eos_token_id, list) and len(self.tokenizer.eos_token_id) > 0:
                self.logger.info(f"Setting pad_token_id to first eos_token_id: {self.tokenizer.eos_token_id[0]}")
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id[0]
        
        # Load model configuration
        self.logger.info(f"Loading model configuration from {model_id}")
        self.config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        
        # Prepare device map
        device_map = None
        
        if device == "cuda" and torch.cuda.is_available():
            device_map = device
        elif device == "cpu":
            device_map = device
        else:
            self.logger.warning(f"Device {device} not available, falling back to CPU")
            device_map = "cpu"
            self.device = "cpu"
        
        try:
            # Load the model with appropriate quantization
            self.logger.info(f"Loading model from {model_id} with {quantization} quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                load_in_4bit=(quantization == "4bit"),
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch.float16,
                max_memory=max_memory,
                offload_folder=offload_folder,
                cache_dir=cache_dir,
            )
        except Exception as e:
            self.logger.warning(f"Error loading model with device mapping: {e}")
            self.logger.info("Falling back to standard loading without device mapping")
            
            # Fallback to standard loading without device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_4bit=(quantization == "4bit"),
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
            
            # Move model to device manually
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
        
        # Enable flash attention if requested and available
        if use_flash_attention and hasattr(self.model.config, "attn_implementation"):
            self.model.config.attn_implementation = "flash_attention_2"
            self.logger.info("Using Flash Attention 2 for faster inference")
        
        # Set up generation config
        if self.tokenizer.pad_token_id is None and self.model.config.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Replace linear layers with quantized versions if needed
        if quantization in ["4bit", "8bit"]:
            self._replace_linear_layers()
        
        # Replace attention layers with optimized versions
        self._replace_attention_layers()
        
        # Calculate optimal batch size if dynamic batching is enabled
        if dynamic_batch_size and self.device == "cuda":
            self.optimal_batch_size = calculate_optimal_batch_size(
                self.model,
                safety_margin=safety_margin
            )
            self.logger.info(f"Calculated optimal batch size: {self.optimal_batch_size}")
        else:
            self.optimal_batch_size = 1
    
    def _replace_linear_layers(self):
        """
        Replace nn.Linear layers with our custom quantized linear layers.
        """
        self.logger.info(f"Replacing linear layers with PyTorch {self.quantization} linear layers")
        
        # Count of replaced layers
        replaced_count = 0
        
        # Choose the appropriate linear layer class
        LinearClass = Linear4Bit if self.quantization == "4bit" else Linear8Bit
        
        # Recursively replace linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get the parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                
                # Create a new quantized linear layer
                try:
                    quantized_linear = LinearClass.from_float(module)
                    
                    # Replace the linear layer
                    setattr(parent, child_name, quantized_linear)
                    replaced_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to replace layer {name}: {e}")
        
        self.logger.info(f"Replaced {replaced_count} linear layers with PyTorch {self.quantization} linear layers")
    
    def _replace_attention_layers(self):
        """
        Replace attention layers with optimized versions.
        """
        self.logger.info("Replacing attention layers with optimized versions")
        
        # Count of replaced layers
        replaced_count = 0
        
        # Recursively replace attention layers
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.MultiheadAttention, torch.nn.Attention)):
                # Get the parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                
                # Get attention parameters
                num_heads = module.num_heads
                head_dim = module.head_dim if hasattr(module, "head_dim") else module.embed_dim // num_heads
                dropout = module.dropout.p if isinstance(module.dropout, torch.nn.Dropout) else 0.0
                
                # Create optimized attention layer
                try:
                    optimized_attention = OptimizedAttention(
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        use_flash_attention=self.use_flash_attention,
                        causal=True  # Most LLMs use causal attention
                    )
                    
                    # Copy weights from original attention layer
                    if hasattr(module, "in_proj_weight"):
                        q_proj, k_proj, v_proj = module.in_proj_weight.chunk(3, dim=0)
                        optimized_attention.q_proj.weight.data = q_proj
                        optimized_attention.k_proj.weight.data = k_proj
                        optimized_attention.v_proj.weight.data = v_proj
                    
                    if hasattr(module, "in_proj_bias"):
                        q_bias, k_bias, v_bias = module.in_proj_bias.chunk(3, dim=0)
                        optimized_attention.q_proj.bias.data = q_bias
                        optimized_attention.k_proj.bias.data = k_bias
                        optimized_attention.v_proj.bias.data = v_bias
                    
                    if hasattr(module, "out_proj"):
                        optimized_attention.out_proj.weight.data = module.out_proj.weight.data
                        optimized_attention.out_proj.bias.data = module.out_proj.bias.data
                    
                    # Replace the attention layer
                    setattr(parent, child_name, optimized_attention)
                    replaced_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to replace attention layer {name}: {e}")
        
        self.logger.info(f"Replaced {replaced_count} attention layers with optimized versions")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt or list of prompts
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text or list of generated texts
        """
        # Handle single prompt or list of prompts
        is_single_prompt = isinstance(prompt, str)
        prompts = [prompt] if is_single_prompt else prompt
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Reshape outputs if multiple sequences per prompt
        if num_return_sequences > 1:
            decoded_outputs = [
                decoded_outputs[i:i+num_return_sequences]
                for i in range(0, len(decoded_outputs), num_return_sequences)
            ]
        
        # Return single output or list of outputs
        if is_single_prompt and num_return_sequences == 1:
            return decoded_outputs[0]
        return decoded_outputs
    
    def embed(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Get embeddings for text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Tensor of embeddings
        """
        # Handle single text or list of texts
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model.model(**inputs, output_hidden_states=True)
            # Use the last hidden state of the last token as the embedding
            embeddings = outputs.hidden_states[-1][:, -1, :]
        
        return embeddings
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for a batch of prompts with dynamic batch sizing.
        
        Args:
            prompts: List of input prompts
            batch_size: Optional batch size override
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated texts
        """
        # Use dynamic batch size if enabled and no override provided
        if batch_size is None and self.dynamic_batch_size:
            batch_size = self.optimal_batch_size
        
        results = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_results = self.generate(batch_prompts, **kwargs)
            results.extend(batch_results)
            
            # Clear cache between batches to prevent memory buildup
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return results
    
    def get_memory_usage(self):
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage statistics
        """
        return get_memory_usage()
    
    def save_memory_snapshot(self, filename: str = "memory_snapshot.txt"):
        """
        Save a snapshot of memory usage.
        
        Args:
            filename: Name of the file to save the snapshot to
        """
        if self.device == "cuda":
            # Get current memory usage
            current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
            # Save to file
            with open(filename, "w") as f:
                f.write(f"Current memory usage: {current_memory:.2f} GB\n")
                f.write(f"Maximum memory usage: {max_memory:.2f} GB\n")
                
                # Log model size
                f.write("\nModel size breakdown:\n")
                total_params = 0
                for name, param in self.model.named_parameters():
                    param_size = param.numel() * param.element_size() / (1024 ** 2)  # MB
                    f.write(f"{name}: {param_size:.2f} MB\n")
                    total_params += param.numel()
                
                f.write(f"\nTotal parameters: {total_params:,}\n")
            
            self.logger.info(f"Memory snapshot saved to {filename}")
        else:
            self.logger.warning("Memory snapshot only available for CUDA devices")
    
    def clear_cache(self):
        """
        Clear CUDA cache to free up memory.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")
        else:
            self.logger.warning("Cache clearing only available for CUDA devices")
    
    def _check_bnb_compatibility(self):
        """
        Check if bitsandbytes is compatible with the current environment.
        
        Returns:
            bool: True if bitsandbytes is compatible, False otherwise
        """
        try:
            import bitsandbytes
            # Check if the version is compatible
            version = getattr(bitsandbytes, "__version__", "0.0.0")
            major, minor, patch = map(int, version.split(".")[:3])
            
            # Require at least version 0.41.0
            if major > 0 or (major == 0 and minor >= 41):
                return True
            else:
                self.logger.warning(f"bitsandbytes version {version} is too old, need at least 0.41.0")
                return False
        except ImportError:
            self.logger.warning("bitsandbytes not installed")
            return False
        except Exception as e:
            self.logger.warning(f"Error checking bitsandbytes compatibility: {e}")
            return False 