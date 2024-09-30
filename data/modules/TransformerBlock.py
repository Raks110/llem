import torch.nn as nn
from modules.FeedForward import FeedForward
from modules.LayerNorm import LayerNorm
from modules.MultiHeadAttention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.embedding_dimension)
        self.norm2 = LayerNorm(config.embedding_dimension)
        self.drop_shortcut = nn.Dropout(config.dropout_rate)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
