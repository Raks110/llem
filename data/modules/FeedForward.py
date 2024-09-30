import torch.nn as nn
from modules.GELU import GELU


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                config.embedding_dimension,
                config.feed_forward_output_scale * config.embedding_dimension
            ),
            GELU(),
            nn.Linear(
                config.feed_forward_output_scale * config.embedding_dimension,
                config.embedding_dimension
            )
        )

    def forward(self, x):
        return self.layers(x)
