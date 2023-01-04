import torch
import torch.nn as nn
from utils import device


class MemNet(nn.Module):
    def __init__(
        self,
        input_dim: int = None,
        embed_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()
        self.device = device()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, output_dim),
            nn.Tanh(),
        ).to(self.device)

    def forward(self, obs, state=None):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        logits = self.net(obs)

        return logits, state
