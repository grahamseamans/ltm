import torch
import torch.nn as nn
from utils import device

from search_attn import MultiHeadAttentionSearch
from memories import Memories


"""
mem net could be:
    encoder -> 64 vec or something
    that same encoder is used by the memnet to encodde the memories

    could then just have it literally do a knn search and get the 6 most similar vecs?

    no because the memories are more than just obs...

    but we could have it just find similar obs honestly, I don't see why not...
    and then the network gets back all of these memories as a big vector?

    we could have an encoder here that's really just a multiheaded attention layer

    and we could have the obs be encoded , so it's really just (encoded obs, act, ret)

    then that really easily feeds into the same multiheaded attn code?

    I kinda want each thinking step to get in:
        similar experiences (obs) (10?)

    dumbest possible way to do this:
        takes in an obs
        get back the actions and returns associated with those actions
        then what does it do?
            then it searches those? (query = obs, key = mem, value = mem)
            just gets action with max return
            maybe both is best?
        only get's back (action, return) so no need to make obs encoded (nice)

    still dreams the same way on ret action obs tuples


"""


class MemNet(nn.Module):
    def __init__(
        self,
        # feature_dim: int,
        mem: Memories,
    ):
        super().__init__()
        self.embed_dim = 64
        num_heads = 4
        self.k = 10
        # obs_len = observation_space.shape[-1]
        # rew_len = 2/
        # act_len = action_space.n
        search_output_dim = self.embed_dim * num_heads
        self.output_dim = 64

        self.memory = mem
        # self.memory = Memories(
        #     obs_len=feature_dim, rew_len=rew_len, act_len=act_len, mem_thresh=128
        # )
        mem_len = mem.act_len + mem.ret_len

        self.obs_embed = nn.Linear(self.memory.obs_len, self.embed_dim).to(device())

        # self.search_attn = MultiHeadAttentionSearch(
        #     embedding_dim=self.embed_dim,
        #     num_heads=search_heads,
        #     q_features=obs_len,
        #     mem_features=mem_len,
        # ).to(device())
        self.attn_1 = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            kdim=mem_len,
            vdim=mem_len,
            batch_first=True,
        ).to(device())
        # self.attn_2 = nn.MultiheadAttention(
        #     embed_dim=self.embed_dim,
        #     num_heads=num_heads,
        #     kdim=mem_len,
        #     vdim=mem_len,
        #     batch_first=True,
        # ).to(device())
        self.out_embed = nn.Linear(search_output_dim, self.output_dim).to(device())

        # feat_len = 10  # output of both attns concatenated...

    def forward(self, obs, state=None):
        obs = torch.as_tensor(obs, device=device(), dtype=torch.float32)
        # mem = self.memory.memories

        mem = self.memory.memories[:, : self.memory.obs_len]
        """
        shape 
            mems = 256, 111
            obs = 10, 111
        """

        batch_size, obs_len = obs.shape
        num_mems, mem_dim = mem.shape
        obs = torch.unsqueeze(obs, dim=1)
        mem = mem.expand(batch_size, num_mems, mem_dim)

        dist = torch.norm(mem - obs, dim=1, p=None)
        knn = dist.topk(self.k, largest=False)

        # print("kNN dist: {}, index: {}".format(knn.values, knn.indices))

        mems = self.memory.memories[knn.indices, self.memory.obs_len :]

        # print(search.shape)
        # print(mem.shape)

        obs_embeded = self.obs_embed(obs)
        features, attn_weights = self.attn_1(obs_embeded, mems, mems)
        # features, attn_weights = self.attn_2(features, mem, mem)
        # print(features.shape)
        searched = torch.flatten(features, start_dim=1)
        logits = self.out_embed(searched)

        # print('feature.shape ', features.shape, 'out len', self.out_len)
        return logits, state

    # def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    #     features = self.mem_stuff(features)
    #     return super().forward(features)

    # def forward_actor(self, features: th.Tensor) -> th.Tensor:
    #     features = self.mem_stuff(features)
    #     return super().forward_actor(features)

    # def forward_critic(self, features: th.Tensor) -> th.Tensor:
    #     features = self.mem_stuff(features)
    #     return super().forward_critic(features)

    # def prod(x):
    #     return reduce(operator.mul, x, 1)
