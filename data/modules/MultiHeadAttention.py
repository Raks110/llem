import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dimension % config.num_attention_heads == 0, \
            "Embedding dimension must be divisible by number of Attention heads"

        self.in_dimension = config.embedding_dimension
        self.out_dimension = config.embedding_dimension
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.embedding_dimension // config.num_attention_heads

        self.W_query = nn.Linear(
            self.in_dimension,
            self.out_dimension,
            bias=config.qkv_bias
        )
        self.W_key = nn.Linear(
            self.in_dimension,
            self.out_dimension,
            bias=config.qkv_bias
        )
        self.W_value = nn.Linear(
            self.in_dimension,
            self.out_dimension,
            bias=config.qkv_bias
        )
        self.out_proj = nn.Linear(
            self.in_dimension,
            self.out_dimension,
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones(
                    config.context_length,
                    config.context_length
                ),
                diagonal=1
            )
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
