"""
Custom PyTorch kernels for efficient inference with quantized models.
Provides optimized implementations for 4-bit and 8-bit quantized inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile") and torch.__version__ >= "2.0.0"

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class Linear4Bit(nn.Module):
    """
    Linear layer using 4-bit quantized weights with PyTorch.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.packed_weight = nn.Parameter(
            torch.zeros((in_features // 2 + (in_features % 2), out_features), dtype=torch.uint8, device=device),
            requires_grad=False
        )
        
        self.quant_block_size = 32
        num_blocks = math.ceil(in_features / self.quant_block_size)
        self.scales = nn.Parameter(
            torch.ones((num_blocks, out_features), dtype=torch.float16, device=device),
            requires_grad=False
        )
        self.zeros = nn.Parameter(
            torch.zeros((num_blocks, out_features), dtype=torch.float16, device=device),
            requires_grad=False
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter('bias', None)
        
        self._optimized_forward = None
        
        self.register_buffer('lookup_table', torch.arange(16, dtype=torch.float16, device=device))
        
        if TORCH_COMPILE_AVAILABLE:
            self._setup_compiled_forward()
    
    def _setup_compiled_forward(self):
        if TORCH_COMPILE_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self._optimized_forward = torch.compile(self._bnb_4bit_matmul_optimized)
                    self._optimized_lookup = torch.compile(self._bnb_4bit_matmul_lookup)
                else:
                    self._optimized_forward = torch.compile(self._bnb_4bit_matmul_optimized)
            except Exception:
                pass
    
    def forward(self, x):
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.reshape(-1, self.in_features)
        
        if self._optimized_forward is not None:
            output = self._optimized_forward(x)
        else:
            if x.shape[0] >= 8:
                output = self._bnb_4bit_matmul_batched(x)
            elif x.is_cuda and x.shape[0] < 4:
                output = self._bnb_4bit_matmul_lookup(x)
            else:
                output = self._bnb_4bit_matmul_optimized(x)
        
        if self.bias is not None:
            output += self.bias
        
        if len(orig_shape) > 2:
            output = output.reshape(*orig_shape[:-1], self.out_features)
        
        return output
    
    def _bnb_4bit_matmul_batched(self, a):
        M, K = a.shape
        N = self.out_features
        
        c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
        
        for block_idx in range(math.ceil(K / self.quant_block_size)):
            start_idx = block_idx * self.quant_block_size
            end_idx = min(start_idx + self.quant_block_size, K)
            block_size = end_idx - start_idx
            
            a_block = a[:, start_idx:end_idx]
            
            scale = self.scales[block_idx]
            zero = self.zeros[block_idx]
            
            w_block = torch.zeros((block_size, N), device=a.device, dtype=torch.float16)
            
            for i in range(0, block_size, 2):
                if start_idx + i >= K:
                    break
                
                byte_idx = (start_idx + i) // 2
                if byte_idx >= self.packed_weight.shape[0]:
                    break
                
                packed_byte = self.packed_weight[byte_idx]
                
                low_nibble = packed_byte & 0xF
                high_nibble = (packed_byte >> 4) & 0xF
                
                if i < block_size:
                    w_block[i] = scale * (low_nibble.to(torch.float16) - zero)
                
                if i + 1 < block_size and start_idx + i + 1 < K:
                    w_block[i + 1] = scale * (high_nibble.to(torch.float16) - zero)
            
            c += torch.matmul(a_block, w_block)
        
        return c
    
    def _bnb_4bit_matmul_optimized(self, a):
        M, K = a.shape
        N = self.out_features
        
        c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
        
        for block_idx in range(math.ceil(K / self.quant_block_size)):
            start_idx = block_idx * self.quant_block_size
            end_idx = min(start_idx + self.quant_block_size, K)
            block_size = end_idx - start_idx
            
            a_block = a[:, start_idx:end_idx]
            
            scale = self.scales[block_idx].unsqueeze(0)
            zero = self.zeros[block_idx].unsqueeze(0)
            
            full_bytes = block_size // 2
            remainder = block_size % 2
            
            for byte_idx in range(full_bytes):
                i = byte_idx * 2
                if start_idx + i >= K:
                    break
                
                packed_byte = self.packed_weight[(start_idx + i) // 2]
                
                low_nibble = packed_byte & 0xF
                high_nibble = (packed_byte >> 4) & 0xF
                
                w_low = scale * (low_nibble.to(torch.float16) - zero)
                w_high = scale * (high_nibble.to(torch.float16) - zero)
                
                a_pair = a_block[:, i:i+2]
                if a_pair.shape[1] == 2:
                    a_low = a_pair[:, 0:1]
                    a_high = a_pair[:, 1:2]
                    
                    c += torch.matmul(a_low, w_low)
                    c += torch.matmul(a_high, w_high)
                else:
                    a_low = a_pair[:, 0:1]
                    c += torch.matmul(a_low, w_low)
            
            if remainder == 1 and full_bytes * 2 < block_size:
                i = full_bytes * 2
                if start_idx + i < K:
                    packed_byte = self.packed_weight[(start_idx + i) // 2]
                    
                    low_nibble = packed_byte & 0xF
                    
                    w_low = scale * (low_nibble.to(torch.float16) - zero)
                    
                    a_low = a_block[:, i:i+1]
                    c += torch.matmul(a_low, w_low)
        
        return c
    
    def _bnb_4bit_matmul_lookup(self, a):
        M, K = a.shape
        N = self.out_features
        
        c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
        
        for block_idx in range(math.ceil(K / self.quant_block_size)):
            start_idx = block_idx * self.quant_block_size
            end_idx = min(start_idx + self.quant_block_size, K)
            block_size = end_idx - start_idx
            
            a_block = a[:, start_idx:end_idx]
            
            scale = self.scales[block_idx]
            zero = self.zeros[block_idx]
            
            lookup = (scale.unsqueeze(0) * (self.lookup_table.unsqueeze(1) - zero.unsqueeze(0))).contiguous()
            
            bytes_to_process = (block_size + 1) // 2
            for byte_offset in range(0, bytes_to_process, 4):
                max_bytes = min(4, bytes_to_process - byte_offset)
                for b in range(max_bytes):
                    byte_idx = (start_idx + (byte_offset + b) * 2) // 2
                    if byte_idx >= self.packed_weight.shape[0]:
                        continue
                    
                    packed_byte = self.packed_weight[byte_idx]
                    
                    low_nibble = packed_byte & 0xF
                    high_nibble = (packed_byte >> 4) & 0xF
                    
                    i = (byte_offset + b) * 2
                    
                    if i < block_size:
                        w_low = torch.gather(lookup, 0, low_nibble.unsqueeze(0).to(torch.int64)).squeeze(0)
                        if i < a_block.shape[1]:
                            c += torch.matmul(a_block[:, i:i+1], w_low.unsqueeze(0))
                    
                    if i + 1 < block_size and i + 1 < a_block.shape[1]:
                        w_high = torch.gather(lookup, 0, high_nibble.unsqueeze(0).to(torch.int64)).squeeze(0)
                        c += torch.matmul(a_block[:, i+1:i+2], w_high.unsqueeze(0))
        
        return c
    
    @classmethod
    def from_float(cls, float_linear, quant_block_size=32):
        """
        Convert a regular nn.Linear to a 4-bit quantized version.
        
        Args:
            float_linear: Regular nn.Linear layer
            quant_block_size: Size of quantization blocks
            
        Returns:
            Quantized linear layer
        """
        device = float_linear.weight.device
        in_features, out_features = float_linear.in_features, float_linear.out_features
        
        quantized = cls(in_features, out_features, 
                        bias=float_linear.bias is not None,
                        device=device)
        
        weight = float_linear.weight.data.t()
        
        num_blocks = math.ceil(in_features / quant_block_size)
        for block_idx in range(num_blocks):
            start_idx = block_idx * quant_block_size
            end_idx = min(start_idx + quant_block_size, in_features)
            if start_idx >= in_features:
                break
                
            block = weight[start_idx:end_idx]
            
            w_min = block.min(dim=0)[0]
            w_max = block.max(dim=0)[0]
            
            scale = (w_max - w_min) / 15
            zero = w_min / scale
            
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero = torch.where(torch.isnan(zero) | torch.isinf(zero), torch.zeros_like(zero), zero)
            
            if block_idx < quantized.scales.shape[0]:
                quantized.scales[block_idx] = scale
                quantized.zeros[block_idx] = zero
        
        packed_weight = torch.zeros((in_features // 2 + (in_features % 2), out_features), 
                                   dtype=torch.uint8, device=device)
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * quant_block_size
            end_idx = min(start_idx + quant_block_size, in_features)
            if start_idx >= in_features:
                break
                
            block = weight[start_idx:end_idx]
            
            scale = quantized.scales[block_idx]
            zero = quantized.zeros[block_idx]
            
            quant_block = torch.clamp(torch.round((block / scale.unsqueeze(0)) + zero.unsqueeze(0)), 0, 15).to(torch.uint8)
            
            for i in range(0, end_idx - start_idx, 2):
                if start_idx + i >= in_features:
                    break
                    
                low_bits = quant_block[i] if i < quant_block.shape[0] else torch.zeros_like(quant_block[0])
                high_bits = quant_block[i+1] if i+1 < quant_block.shape[0] else torch.zeros_like(quant_block[0])
                
                packed = low_bits | (high_bits << 4)
                byte_idx = (start_idx + i) // 2
                if byte_idx < packed_weight.shape[0]:
                    packed_weight[byte_idx] = packed
        
        quantized.packed_weight = nn.Parameter(packed_weight, requires_grad=False)
        
        if float_linear.bias is not None:
            quantized.bias = nn.Parameter(float_linear.bias.clone().to(torch.float16))
        
        return quantized

class QuantizedAttention(nn.Module):
    """
    Memory-efficient attention implementation using 4-bit quantized weights.
    """
    def __init__(self, hidden_size, num_heads, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.q_proj = Linear4Bit(hidden_size, hidden_size)
        self.k_proj = Linear4Bit(hidden_size, hidden_size)
        self.v_proj = Linear4Bit(hidden_size, hidden_size)
        self.o_proj = Linear4Bit(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.scale = 1.0 / math.sqrt(self.head_size)
        
        self._query_states_buffer = None
        self._key_states_buffer = None
        self._value_states_buffer = None
        
        self._optimized_attention = None
        if TORCH_COMPILE_AVAILABLE:
            try:
                self._optimized_attention = torch.compile(self._compute_attention, fullgraph=True)
            except Exception:
                pass
    
    def _compute_attention(self, q, k, v, attention_mask=None):
        """
        Compute attention scores and context.
        Separated for potential compilation.
        """
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None:
            if attention_mask.dtype != attention_scores.dtype:
                attention_mask = attention_mask.to(attention_scores.dtype)
            attention_scores = attention_scores + attention_mask
        
        attention_scores_dtype = attention_scores.dtype
        attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32)
        
        if attention_probs.dtype != attention_scores_dtype:
            attention_probs = attention_probs.to(attention_scores_dtype)
        
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, v)
        
        return context
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, output_attentions=False):
        batch_size, seq_length = hidden_states.size()[:2]
        
        if self._query_states_buffer is None or self._query_states_buffer.shape[0] != batch_size:
            self._query_states_buffer = None
            self._key_states_buffer = None
            self._value_states_buffer = None
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        
        q = q.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2).contiguous()
        k = k.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2).contiguous()
        v = v.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2).contiguous()
        
        kv_seq_length = seq_length
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if not past_k.is_contiguous():
                past_k = past_k.contiguous()
            if not past_v.is_contiguous():
                past_v = past_v.contiguous()
            
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            kv_seq_length = k.shape[2]  # Update sequence length
        
        current_key_value = (k, v) if past_key_value is not None else None
        
        if FLASH_ATTENTION_AVAILABLE and hidden_states.is_cuda:
            q_flash = q.transpose(1, 2)  # [batch_size, seq_length, num_heads, head_size]
            k_flash = k.transpose(1, 2)  # [batch_size, kv_seq_length, num_heads, head_size]
            v_flash = v.transpose(1, 2)  # [batch_size, kv_seq_length, num_heads, head_size]
            
            try:
                dropout_p = 0.0 if not self.training else self.dropout.p
                context = flash_attn_func(
                    q_flash, k_flash, v_flash, 
                    dropout_p=dropout_p, 
                    softmax_scale=self.scale,
                    causal=False  # Set to True for decoder-only models with causal masking
                )
                
                context = context.reshape(batch_size, seq_length, self.hidden_size)
            except Exception:
                attention_mask_reshaped = None
                if attention_mask is not None:
                    attention_mask_reshaped = attention_mask.view(batch_size, 1, 1, kv_seq_length)
                
                if self._optimized_attention is not None:
                    context = self._optimized_attention(q, k, v, attention_mask_reshaped)
                else:
                    context = self._compute_attention(q, k, v, attention_mask_reshaped)
                
                context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        else:
            attention_mask_reshaped = None
            if attention_mask is not None:
                attention_mask_reshaped = attention_mask.view(batch_size, 1, 1, kv_seq_length)
            
            if self._optimized_attention is not None:
                context = self._optimized_attention(q, k, v, attention_mask_reshaped)
            else:
                context = self._compute_attention(q, k, v, attention_mask_reshaped)
            
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        output = self.o_proj(context)
        
        outputs = (output, current_key_value)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs


class QuantizedMLP(nn.Module):
    """
    Memory-efficient MLP implementation using 4-bit quantized weights.
    """
    def __init__(self, hidden_size, intermediate_size, activation_function="gelu"):
        super().__init__()
        self.gate_proj = Linear4Bit(hidden_size, intermediate_size)
        self.up_proj = Linear4Bit(hidden_size, intermediate_size)
        self.down_proj = Linear4Bit(intermediate_size, hidden_size)
        
        if activation_function == "gelu":
            self.act_fn = F.gelu
        elif activation_function == "relu":
            self.act_fn = F.relu
        elif activation_function == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
        
        self._gate_output = None
        self._up_output = None
        
        self._optimized_forward = None
        if TORCH_COMPILE_AVAILABLE:
            try:
                self._optimized_forward = torch.compile(self._forward_impl, fullgraph=True)
            except Exception:
                pass
    
    def _forward_impl(self, x):
        """
        Implementation of forward pass, separated for potential compilation.
        Using fused operations where possible.
        """
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        activated_gate = self.act_fn(gate_output)
        intermediate = activated_gate * up_output
        
        return self.down_proj(intermediate)
    
    def _fused_forward(self, x):
        """
        Fused version of forward pass that combines operations
        for potential kernel fusion on supported hardware.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if (self._gate_output is None or 
            self._gate_output.shape[0] != batch_size or 
            self._gate_output.shape[1] != seq_len):
            
            self._gate_output = None
            self._up_output = None
        
        gate_proj_output = self.gate_proj(x)
        up_proj_output = self.up_proj(x)
        
        activated_gate = self.act_fn(gate_proj_output)
        
        intermediate = activated_gate * up_proj_output
        
        activated_gate = None  # Help garbage collection
        
        output = self.down_proj(intermediate)
        
        return output
    
    def forward(self, x):
        """
        Forward pass with automatic dispatch to best implementation.
        
        Intelligently selects between three optimized implementations:
        1. Compiled implementation: Uses torch.compile for maximum performance when available
        2. Fused implementation: For very large batches, uses buffer management and strategic memory clearing
        3. Standard implementation: Efficient approach for smaller batch sizes
        
        The selection is based on:
        - Availability of torch.compile
        - Input tensor size
        - Hardware capabilities
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_length, hidden_size)
        """
        if self._optimized_forward is not None:
            return self._optimized_forward(x)
        
        if x.numel() > 1000000:  # >1M elements threshold
            return self._fused_forward(x)
        
        return self._forward_impl(x)


class Linear8Bit(nn.Module):
    """
    Linear layer using 8-bit quantized weights with PyTorch.
    Provides a good balance between memory efficiency and accuracy.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.quantized_weight = nn.Parameter(
            torch.zeros((in_features, out_features), dtype=torch.uint8, device=device),
            requires_grad=False
        )
        
        self.quant_block_size = 32
        num_blocks = math.ceil(in_features / self.quant_block_size)
        self.scales = nn.Parameter(
            torch.ones((num_blocks, out_features), dtype=torch.float16, device=device),
            requires_grad=False
        )
        self.zeros = nn.Parameter(
            torch.zeros((num_blocks, out_features), dtype=torch.float16, device=device),
            requires_grad=False
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter('bias', None)
        
        # Cache for optimized implementation
        self._optimized_forward = None
        
        if TORCH_COMPILE_AVAILABLE:
            self._setup_compiled_forward()
    
    def _setup_compiled_forward(self):
        """Set up compiled forward function if available."""
        if TORCH_COMPILE_AVAILABLE:
            try:
                self._optimized_forward = torch.compile(self._optimized_matmul)
            except Exception:
                pass
    
    def forward(self, x):
        """
        Forward pass through the 8-bit quantized linear layer.
        
        Args:
            x: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        # Reshape input if needed
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.reshape(-1, self.in_features)
        
        if self._optimized_forward is not None:
            output = self._optimized_forward(x)
        else:
            output = self._optimized_matmul(x)
        
        if self.bias is not None:
            output += self.bias
        
        if len(orig_shape) > 2:
            output = output.reshape(*orig_shape[:-1], self.out_features)
        
        return output
    
    def _optimized_matmul(self, a):
        """
        Perform matrix multiplication with 8-bit quantized weights.
        Optimized with vectorized operations and memory access patterns.
        
        Args:
            a: Input tensor of shape (M, K)
            
        Returns:
            Output tensor of shape (M, N)
        """
        # Get dimensions
        M, K = a.shape
        N = self.out_features
        
        # Allocate output
        c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
        
        for block_idx in range(math.ceil(K / self.quant_block_size)):
            start_idx = block_idx * self.quant_block_size
            end_idx = min(start_idx + self.quant_block_size, K)
            block_size = end_idx - start_idx
            
            a_block = a[:, start_idx:end_idx]
            
            scale = self.scales[block_idx].unsqueeze(0)  # [1, out_features]
            zero = self.zeros[block_idx].unsqueeze(0)    # [1, out_features]
            
            w_block = self.quantized_weight[start_idx:end_idx]  # [block_size, out_features]
            
            w_block = scale * (w_block.to(torch.float16) - zero)  # [block_size, out_features]
            
            c += torch.matmul(a_block, w_block)
        
        return c
    
    @classmethod
    def from_float(cls, float_linear, quant_block_size=32):
        """
        Convert a regular nn.Linear to an 8-bit quantized version.
        
        Args:
            float_linear: Regular nn.Linear layer
            quant_block_size: Size of quantization blocks
            
        Returns:
            Quantized linear layer
        """
        device = float_linear.weight.device
        in_features, out_features = float_linear.in_features, float_linear.out_features
        
        # Create new 8-bit linear layer
        quantized = cls(in_features, out_features, 
                        bias=float_linear.bias is not None,
                        device=device)
        
        # Quantize weights to 8-bit
        weight = float_linear.weight.data.t()  # Transpose to [in_features, out_features]
        
        num_blocks = math.ceil(in_features / quant_block_size)
        for block_idx in range(num_blocks):
            start_idx = block_idx * quant_block_size
            end_idx = min(start_idx + quant_block_size, in_features)
            if start_idx >= in_features:
                break
                
            block = weight[start_idx:end_idx]  # [block_size, out_features]
            
            w_min = block.min(dim=0)[0]  # [out_features]
            w_max = block.max(dim=0)[0]  # [out_features]
            
            scale = (w_max - w_min) / 255  # 255 is the range of 8-bit (0-255)
            zero = w_min / scale
            
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero = torch.where(torch.isnan(zero) | torch.isinf(zero), torch.zeros_like(zero), zero)
            
            if block_idx < quantized.scales.shape[0]:
                quantized.scales[block_idx] = scale
                quantized.zeros[block_idx] = zero
        
        quantized_weight = torch.clamp(
            torch.round((weight / quantized.scales.unsqueeze(0)) + quantized.zeros.unsqueeze(0)),
            0, 255
        ).to(torch.uint8)
        
        quantized.quantized_weight = nn.Parameter(quantized_weight, requires_grad=False)
        
        if float_linear.bias is not None:
            quantized.bias = nn.Parameter(float_linear.bias.clone().to(torch.float16))
        
        return quantized 