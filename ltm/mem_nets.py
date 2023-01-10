import torch
import torch.nn as nn
import torch.nn.functional as f
from utils import device
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


class Starter(nn.Module):
    def __init__(
        self,
        kv_dim: int,
        embed_dim: int = 64,
        num_thoughts: int = 4,
    ):
        super().__init__()
        self.starter = [
            nn.MultiheadAttention(
                embed_dim, 1, vdim=kv_dim, kdim=kv_dim, batch_first=True
            ).to(device())
            for _ in range(num_thoughts)
        ]

    def forward(self, q, kv):
        thoughts = [start(q, kv, kv)[0] for start in self.starter]
        return torch.cat(thoughts, dim=1)


class Thinker(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        thinking_steps: int = 2,
    ):
        super().__init__()

        self.steps = [
            (
                nn.LayerNorm(embed_dim).to(device()),
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    batch_first=True,
                ).to(device()),
                nn.Tanh(),
            )
            for _ in range(thinking_steps)
        ]

    def forward(self, x):
        for norm, attn, activation in self.steps:
            x = norm(x)
            x, _ = attn(x, x, x)
            x = activation(x)

        return x


class Vanilla(nn.Module):
    def __init__(
        self,
        mem: Memories = None,
        embed_len: int = 64,
        output_len: int = 64,
    ):
        super().__init__()
        self.device = device()
        self.output_dim = output_len
        self.net = nn.Sequential(
            nn.Linear(mem.obs_len, embed_len).to(self.device),
            nn.Tanh(),
            nn.Linear(embed_len, output_len).to(self.device),
            nn.Tanh(),
        )

    def forward(self, obs, state=None):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        logits = self.net(obs)

        return logits, state


class BaseThinker(nn.Module):
    def __init__(
        self,
        mem: Memories,
        k: int = 16,
    ):
        super().__init__()
        self.device = device()
        self.memory = mem
        self.k = k

    def forward(self, obs, state=None):
        return obs

    def preprocess(self, obs):
        obs = torch.as_tensor(obs, device=device(), dtype=torch.float32)
        mem = self.memory.memories

        # reshaping
        batch_size, obs_len = obs.shape
        num_mems, mem_dim = mem.shape
        shaped_obs = torch.unsqueeze(obs, dim=1)
        mem = mem.expand(batch_size, num_mems, mem_dim)

        # topk
        # obs_mem = mem[:, :, : self.memory.obs_len]
        # sim_mems = self.topk(obs_mem, shaped_obs, self.k)
        # best = self.best_acts(sim_mems)

        return obs, shaped_obs, mem

    def topk(self, mems, obs, k):
        obs = f.normalize(obs)
        dist = torch.norm(mems - obs, dim=1, p=None)
        knn = dist.topk(k, largest=False)
        return self.memory.memories[knn.indices, :]

    def best_acts(self, sim_mems):
        sim_act_ret = sim_mems[:, self.memory.obs_len :]
        sim_act = sim_act_ret[..., : self.memory.act_len]  # 10 16 6
        sim_ret = sim_act_ret[..., self.memory.act_len :]  # 10 16 2
        sim_ret = torch.einsum("ijk->ij", sim_ret)  # 10 16
        maxes = torch.argmax(sim_ret, dim=1)  # 10
        selector = torch.zeros(sim_act.shape[:-1]).bool()
        for i, x in enumerate(maxes):
            selector[i, x] = True
        best_acts = sim_act[selector, :]  # should be 10 6
        return best_acts


class Hybrid(BaseThinker):
    def __init__(
        self,
        mem: Memories,
        num_heads: int = 4,
        embed_len: int = 64,
        k: int = 16,
        output_len: int = 64,
    ):
        super().__init__(mem, k)
        self.output_dim = output_len
        self.activation = nn.Tanh()
        self.obs_embed = nn.Linear(self.memory.obs_len, embed_len)
        self.mem_search = nn.MultiheadAttention(
            embed_len,
            num_heads,
            kdim=mem.mem_len,
            vdim=mem.act_len,
            batch_first=True,
        )
        self.instinct = nn.Sequential(
            nn.Linear(mem.obs_len, embed_len).to(self.device),
            nn.Tanh(),
            nn.Linear(embed_len, embed_len).to(self.device),
            nn.Tanh(),
        )
        self.decide = nn.Linear(embed_len * 2, 2)
        self.out_embed = nn.Linear(embed_len, output_len)

        self.attn_decide = nn.MultiheadAttention(
            embed_len,
            num_heads=4,
            batch_first=True,
        )

        self.descision_type = False

        self.to(device())

    def forward(self, obs, state=None):
        obs, shaped_obs, mems = self.preprocess(obs)
        acts = mems[
            :, :, self.memory.obs_len : self.memory.obs_len + self.memory.act_len
        ]
        q = self.activation(self.obs_embed(shaped_obs))
        remembered = self.activation(self.mem_search(q, mems, acts)[0])

        reflex = self.instinct(obs)

        if self.descision_type:
            remembered = torch.squeeze(remembered, dim=1)
            combo = torch.cat([remembered, reflex], dim=1)
            descision = f.softmax(self.decide(combo), dim=1)
            out = remembered * descision[:, 0] + reflex * descision[:, 1]
        else:
            reflex = torch.unsqueeze(reflex, dim=1)
            combo = torch.cat([remembered, reflex], dim=1)
            out = torch.sum(self.attn_decide(combo, combo, combo)[0], dim=1)
            out = self.activation(out)

        return out, state


class SingleThought(BaseThinker):
    def __init__(
        self,
        mem: Memories,
        num_heads: int = 4,
        embed_len: int = 64,
        k: int = 16,
        output_len: int = 64,
    ):
        super().__init__(mem, k)
        self.output_dim = output_len
        self.activation = nn.Tanh()
        self.obs_embed = nn.Linear(self.memory.obs_len, embed_len)
        self.thought_1 = Thinker(self.memory.mem_len, num_heads, embed_len)
        self.out_embed = nn.Linear(embed_len, output_len)

    def forward(self, obs, state=None):
        shaped_obs, sim_mems = self.preprocess(obs)
        features = self.activation(self.obs_embed(shaped_obs))
        thought = self.thought_1(features, sim_mems)
        thought = torch.squeeze(thought, dim=1)
        out = self.activation(self.out_embed(thought))
        return out, state


class MultiThought(BaseThinker):
    def __init__(
        self,
        mem: Memories,
        num_heads: int = 4,
        embed_len: int = 32,
        k: int = 16,
        output_len: int = 64,
        num_thoughts: int = 4,
        thinking_steps: int = 4,
    ):
        super().__init__(mem, k)
        self.output_dim = output_len
        self.activation = nn.Tanh()
        self.obs_embed = nn.Linear(self.memory.obs_len, embed_len)
        self.starter = Starter(self.memory.mem_len, embed_len, num_thoughts)
        self.thinker = Thinker(embed_len, num_heads, thinking_steps)
        self.out_embed = nn.Linear(embed_len * num_thoughts, output_len)

    def forward(self, obs, state=None):
        shaped_obs, sim_mems = self.preprocess(obs)
        features = self.activation(self.obs_embed(shaped_obs))
        started = self.starter(features, sim_mems)
        thought = self.thinker(started)
        thought = torch.flatten(thought, 1)
        out = self.activation(self.out_embed(thought))
        return out, state


# class MemNet(nn.Module):
#     def __init__(
#         self,
#         mem: Memories,
#         num_heads: int = 4,
#         embed_dim: int = 64,
#         k: int = 16,
#         output_dim: int = 64,
#     ):
#         super().__init__()
#         self.memory = mem

#         self.output_dim = output_dim
#         self.k = k
#         mem_len = mem.act_len + mem.ret_len

#         self.activation = nn.Tanh()
#         self.obs_embed = nn.Linear(self.memory.obs_len, embed_dim).to(device())
#         self.attn_1 = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             kdim=mem_len,
#             vdim=mem_len,
#             batch_first=True,
#         ).to(device())
#         # output of attn flattened is just embed dim because there's only one
#         # obs that's being put into the attn layer for each part of the batch
#         self.out_embed = nn.Linear(embed_dim, output_dim).to(device())
#         self.best_act_out = nn.Linear(embed_dim + mem.act_len, output_dim).to(device())

#     def forward(self, obs, state=None):
#         obs = torch.as_tensor(obs, device=device(), dtype=torch.float32)

#         # reshaping
#         obs_mem = self.memory.memories[:, : self.memory.obs_len]
#         batch_size, obs_len = obs.shape
#         num_mems, mem_dim = obs_mem.shape
#         obs = torch.unsqueeze(obs, dim=1)
#         obs_mem = obs_mem.expand(batch_size, num_mems, mem_dim)

#         # topk.
#         obs_mem_normed = f.normalize(obs_mem)
#         dist = torch.norm(obs_mem_normed - obs, dim=1, p=None)
#         knn = dist.topk(self.k, largest=False)
#         sim_mems = self.memory.memories[knn.indices, :]
#         sim_act_ret = sim_mems[:, self.memory.obs_len :]

#         # best action
#         sim_act = sim_act_ret[..., : self.memory.act_len]  # 10 16 6
#         sim_ret = sim_act_ret[..., self.memory.act_len :]  # 10 16 2
#         sim_ret = torch.einsum("ijk->ij", sim_ret)  # 10 16
#         maxes = torch.argmax(sim_ret, dim=1)  # 10
#         selector = torch.zeros(sim_act.shape[:-1]).bool()
#         for i, x in enumerate(maxes):
#             selector[i, x] = True
#         best_acts = sim_act[selector, :]  # should be 10 6

#         # actual layers
#         obs_embeded = self.activation(self.obs_embed(torch.squeeze(obs, 1)))  # 10 64
#         embed_best = torch.cat([obs_embeded, best_acts], dim=1)
#         logits = self.activation(self.best_act_out(embed_best))
#         # mems = self.activation(mems)
#         # features, attn_weights = self.attn_1(obs_embeded, mems, mems)  # 10 1 64
#         # features = self.activation(features)
#         # searched = torch.flatten(features, start_dim=1)
#         # logits = self.out_embed(searched)
#         # logits = self.out_embed(searched)

#         return logits, state

#     # def forward(self, obs, state=None):
#     #     obs = torch.as_tensor(obs, device=device(), dtype=torch.float32)

#     #     # reshaping
#     #     mem = self.memory.memories[:, : self.memory.obs_len]
#     #     batch_size, obs_len = obs.shape
#     #     num_mems, mem_dim = mem.shape
#     #     obs = torch.unsqueeze(obs, dim=1)
#     #     mem = mem.expand(batch_size, num_mems, mem_dim)

#     #     # topk
#     #     dist = torch.norm(mem - obs, dim=1, p=None)
#     #     knn = dist.topk(self.k, largest=False)
#     #     mems = self.memory.memories[knn.indices, self.memory.obs_len :]  # 10 10 10

#     #     # actual layers
#     #     obs_embeded = self.obs_embed(obs)  # 10 1 64
#     #     obs_embeded = self.activation(obs_embeded)
#     #     mems = self.activation(mems)
#     #     features, attn_weights = self.attn_1(obs_embeded, mems, mems)  # 10 1 64
#     #     features = self.activation(features)
#     #     searched = torch.flatten(features, start_dim=1)
#     #     logits = self.out_embed(searched)

#     #     return logits, state
