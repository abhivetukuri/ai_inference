"""
Attention optimizations for efficient inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

def _flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention 2 forward pass.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        key: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        value: Value tensor of shape (batch_size, seq_len, num_heads, head_dim)
        mask: Optional attention mask
        dropout_p: Dropout probability
        causal: Whether to use causal attention
        softmax_scale: Optional scaling factor for softmax
        
    Returns:
        Output tensor of shape (batch_size, seq_len, num_heads, head_dim)
    """
    if not torch.cuda.is_available():
        return _standard_attention_forward(
            query, key, value, mask, dropout_p, causal, softmax_scale
        )
    
    # Reshape for flash attention
    batch_size, seq_len, num_heads, head_dim = query.shape
    q = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    k = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    v = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    
    # Use flash attention if available
    try:
        from flash_attn import flash_attn_func
        output = flash_attn_func(
            q, k, v,
            mask=mask,
            dropout_p=dropout_p,
            causal=causal,
            softmax_scale=softmax_scale
        )
    except ImportError:
        # Fallback to standard attention if flash attention is not available
        output = _standard_attention_forward(
            query, key, value, mask, dropout_p, causal, softmax_scale
        )
    
    # Reshape back
    output = output.reshape(batch_size, num_heads, seq_len, head_dim)
    return output.transpose(1, 2)

def _standard_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard attention forward pass with optimizations.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        key: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        value: Value tensor of shape (batch_size, seq_len, num_heads, head_dim)
        mask: Optional attention mask
        dropout_p: Dropout probability
        causal: Whether to use causal attention
        softmax_scale: Optional scaling factor for softmax
        
    Returns:
        Output tensor of shape (batch_size, seq_len, num_heads, head_dim)
    """
    batch_size, seq_len, num_heads, head_dim = query.shape
    
    # Scale query
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply causal mask if requested
    if causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(scores.device)
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply softmax
    attn = F.softmax(scores, dim=-1)
    
    # Apply dropout
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    
    # Compute output
    output = torch.matmul(attn, value)
    
    return output

class OptimizedAttention(nn.Module):
    """
    Optimized attention module with Flash Attention 2 support.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        causal: bool = False,
    ):
        """
        Initialize the attention module.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            dropout: Dropout probability
            use_flash_attention: Whether to use Flash Attention 2
            causal: Whether to use causal attention
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.causal = causal
        
        # Initialize projection matrices
        self.q_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        
        # Initialize with better defaults
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Set bias to zero
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, hidden_dim)
            key: Optional key tensor of shape (batch_size, seq_len, hidden_dim)
            value: Optional value tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            key_padding_mask: Optional key padding mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = query.shape
        
        # Use query as key and value if not provided
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Project inputs
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        
        # Compute attention
        if self.use_flash_attention:
            output = _flash_attention_forward(
                q, k, v,
                mask=mask,
                dropout_p=self.dropout,
                causal=self.causal
            )
        else:
            output = _standard_attention_forward(
                q, k, v,
                mask=mask,
                dropout_p=self.dropout,
                causal=self.causal
            )
        
        # Reshape and project output
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.out_proj(output)
        
        return output 