import torch
import torch.nn.functional as f
import random
from stable_baselines3.common.utils import get_device


class Memories:
    def __init__(self, obs_len, rew_len, act_len, mem_thresh):
        self.obs_len = obs_len
        self.rew_len = rew_len
        self.act_len = act_len
        self.mem_thresh = mem_thresh
        self.device = get_device("auto")
        self.memories = torch.ones(mem_thresh, obs_len + rew_len + act_len).to(
            self.device
        )
        self.rew_avg = None

    def add(self, obs, rew, act):
        if self.rew_avg == None:
            self.rew_avg = torch.mean(rew)

        rew = rew.repeat(1, 2)
        rew[:, 1] = self.rew_avg

        obs = f.normalize(obs)
        rew = f.normalize(rew)
        act = f.normalize(act)

        new_memories = torch.cat((obs, rew, act), dim=1)
        self.memories = torch.cat((self.memories, new_memories))

        self.dream()

    def dream(self):
        to_remove = len(self.memories) - self.mem_thresh

        if to_remove < 0:
            return

        print(
            f"dreaming with {len(self.memories)} memories and a threshold of {self.mem_thresh}"
        )

        mem_sim = torch.einsum("ij,kj->ik", self.memories, self.memories).to("cpu")
        idxs = self.unravel_indices(
            torch.argsort(mem_sim.flatten(), descending=True), mem_sim.shape
        )
        idxs = idxs[range(len(self.memories), len(idxs), 2)]

        idx_idx = 0
        idx_remove = 0
        remove_set = set()
        while idx_remove < to_remove:
            a, b = idxs[idx_idx]
            idx_idx += 1
            if a not in remove_set and b not in remove_set:
                remove_set.add((a if random.random() > 0.5 else b).item())
                idx_remove += 1

        keep = list(set(range(len(self.memories))) - remove_set)
        self.memories = self.memories[keep]

    def unravel_indices(self, indices, shape):
        coord = []
        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = torch.div(indices, dim, rounding_mode="trunc")
        coord = torch.stack(coord[::-1], dim=-1)
        return coord
