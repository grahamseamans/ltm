import torch
import torch.nn.functional as f
import random
import math
from utils import device
from tianshou.data import Batch


class Memories:
    def __init__(self, obs_len, rew_len, act_len, mem_thresh):
        self.rew_len = 2
        self.obs_len = math.prod(obs_len)
        self.act_len = math.prod(act_len)
        self.mem_thresh = mem_thresh
        total_len = self.obs_len + self.rew_len + self.act_len
        self._memories = torch.ones(0, total_len).to(device())
        self._dummy = torch.rand(self.mem_thresh, total_len).to(device())
        self.mem_empty = True

    @property
    def memories(self):
        return self._dummy if self.mem_empty else self._memories

    def add(self, batch: Batch):
        mems_needed = 2
        if self.mem_empty:
            mems_needed = self.mem_thresh - len(self.memories)
            if len(self.memories) >= self.mem_thresh:
                self.mem_empty = False

        to_add = next(batch.split(min(len(batch), mems_needed), shuffle=True))
        obs = to_add.obs
        rew = to_add.returns
        act = to_add.act

        rew = rew.repeat(1, 2)
        rew[:, 1] = 1

        obs = f.normalize(obs)
        rew = f.normalize(rew)
        act = f.normalize(act)

        new_memories = torch.cat((obs, rew, act), dim=1)
        self._memories = torch.cat((self._memories, new_memories))

        self.dream()

    def dream(self):
        to_remove = len(self._memories) - self.mem_thresh
        if to_remove < 0:
            return
        print(
            f"dreaming with {len(self._memories)} memories and a threshold of {self.mem_thresh}"
        )
        mem_sim = torch.einsum("ij,kj->ik", self._memories, self._memories).to("cpu")
        idxs = self.unravel_indices(
            torch.argsort(mem_sim.flatten(), descending=True), mem_sim.shape
        )
        idxs = idxs[range(len(self._memories), len(idxs), 2)]

        idx_idx = 0
        idx_remove = 0
        remove_set = set()
        while idx_remove < to_remove:
            a, b = idxs[idx_idx]
            idx_idx += 1
            if a not in remove_set and b not in remove_set:
                remove_set.add((a if random.random() > 0.5 else b).item())
                idx_remove += 1

        keep = list(set(range(len(self._memories))) - remove_set)
        self._memories = self._memories[keep]

    def unravel_indices(self, indices, shape):
        coord = []
        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = torch.div(indices, dim, rounding_mode="trunc")
        coord = torch.stack(coord[::-1], dim=-1)
        return coord
