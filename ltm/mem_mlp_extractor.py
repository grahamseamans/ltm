from stable_baselines3.common.torch_layers import MlpExtractor
from typing import Dict, List, Tuple, Type, Union
from stable_baselines3.common.utils import get_device
from functools import reduce
import operator
import gym
import torch as th
import torch.nn as nn

from ltm.search_attn import MultiHeadAttentionSearch
from memories import Memories


class MemoryMlpExtractor(MlpExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        self.embed_dim = 16
        search_heads = 16
        obs_len = observation_space.shape[-1]
        rew_len = 2
        act_len = action_space.n
        out_len = self.embed_dim * search_heads
        # print(f'out_len {out_len}')
        self.out_len = out_len
        super().__init__(out_len, net_arch, activation_fn, device)
        device = get_device("auto")

        self.memory = Memories(
            obs_len=feature_dim, rew_len=rew_len, act_len=act_len, mem_thresh=1_024
        )

        mem = self.memory._memories
        mem_len = mem.shape[1]

        self.search_attn = MultiHeadAttentionSearch(
            embedding_dim=self.embed_dim,
            q_features=obs_len,
            mem_features=mem_len,
            head_num=search_heads,
        ).to(device)
        self.attn_1 = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            kdim=mem_len,
            vdim=mem_len,
            batch_first=True,
        ).to(device)
        self.attn_2 = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            kdim=mem_len,
            vdim=mem_len,
            batch_first=True,
        ).to(device)

        feat_len = 10  # output of both attns concatenated...

    def mem_stuff(self, obs):
        mem = self.memory._memories

        batch_size, obs_len = obs.shape
        num_mems, mem_dim = mem.shape

        obs = th.unsqueeze(obs, dim=1)
        mem = mem.expand(batch_size, num_mems, mem_dim)

        search = self.search_attn(obs, mem, mem)
        # print(search.shape)
        # print(mem.shape)
        features, attn_weights = self.attn_1(search, mem, mem)
        features, attn_weights = self.attn_2(features, mem, mem)
        # print(features.shape)
        features = th.flatten(features, start_dim=1)
        # print('feature.shape ', features.shape, 'out len', self.out_len)
        return features

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.mem_stuff(features)
        return super().forward(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        features = self.mem_stuff(features)
        return super().forward_actor(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features = self.mem_stuff(features)
        return super().forward_critic(features)

    def prod(x):
        return reduce(operator.mul, x, 1)
