from typing import Optional

import torch
from torch import nn


class StyleAttnProcessor(nn.Module):
    """
    Cross-attention processor that injects style tokens (IP-Adapter style).
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.to_k_style = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_style = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.style_tokens: Optional[torch.Tensor] = None

    def set_style_tokens(self, tokens: Optional[torch.Tensor]) -> None:
        self.style_tokens = tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Standard attention
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Style tokens injection
        if self.style_tokens is not None:
            k_style = self.to_k_style(self.style_tokens) * self.scale
            v_style = self.to_v_style(self.style_tokens) * self.scale
            key = torch.cat([key, k_style], dim=1)
            value = torch.cat([value, v_style], dim=1)

        hidden_states = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(hidden_states, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
