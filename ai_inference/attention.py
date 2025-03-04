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
    return _standard_attention_forward(
        query, key, value, mask, dropout_p, causal, softmax_scale
    )

def _standard_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = query.shape
    
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    if causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(scores.device)
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    
    output = torch.matmul(attn, value)
    
    return output

class OptimizedAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.causal = causal
        
        self.q_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
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
        batch_size, seq_len, _ = query.shape
        
        if key is None:
            key = query
        if value is None:
            value = query
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        
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
        
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.out_proj(output)
        
        return output