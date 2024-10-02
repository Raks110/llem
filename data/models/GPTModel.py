import torch
from torch import nn

from modules.LayerNorm import LayerNorm
from modules.TransformerBlock import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dimension)
        self.pos_emb = nn.Embedding(config.context_length, config.embedding_dimension)
        self.drop_emb = nn.Dropout(config.dropout_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)])

        self.final_norm = LayerNorm(config.embedding_dimension)
        self.out_head = nn.Linear(
            config.embedding_dimension, config.vocab_size, bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
