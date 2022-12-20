import torch
import torch.nn.functional as f
import random
import math
from utils import device
from tianshou.data import Batch
import einops
import numpy as np
import scipy
from sklearn.preprocessing import normalize


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
        mems_needed = 16
        if self.mem_empty:
            mems_needed = self.mem_thresh - len(self._memories)
            if len(self._memories) + mems_needed >= self.mem_thresh:
                self.mem_empty = False

        to_add = next(batch.split(mems_needed, shuffle=True))
        obs = to_add.obs
        rew = to_add.returns
        act = to_add.act

        rew = einops.repeat(rew, "m -> m k", k=2)
        rew[:, 1] = 1

        obs = torch.as_tensor(obs, device=device(), dtype=torch.float32)
        rew = torch.as_tensor(rew, device=device(), dtype=torch.float32)
        act = torch.as_tensor(act, device=device(), dtype=torch.float32)

        obs = f.normalize(obs)
        rew = f.normalize(rew)
        act = f.normalize(act)

        new_memories = torch.cat((obs, rew, act), dim=1)
        self._memories = torch.cat((self._memories, new_memories))

        self.dream()

    def dream(self):

        if len(self._memories) <= self.mem_thresh:
            return

        x = self._memories.numpy()
        i = len(x)
        i_thresh = self.mem_thresh

        tree = scipy.spatial.KDTree(x)
        close_ret = tree.query(x, k=(i - i_thresh + 1))
        close = close_ret[0]
        idxs = close_ret[1]

        close = close[:, 1:]
        idxs = idxs[:, 1:]

        flattened = close.flatten()
        sorted = np.argsort(flattened)
        close_shape = close.shape
        unraveled = np.unravel_index(sorted, close_shape)
        unraveled_stack = np.dstack(unraveled)
        ordered = unraveled[0]
        pairs = ordered.reshape((len(ordered) // 2, 2))
        idxs = pairs

        remove = []
        idx = 0
        while len(remove) < i - i_thresh:
            a, b = idxs[idx]
            idx += 1
            if a not in remove and b not in remove:
                remove.append(a if random.random() > 0.5 else b)

        keep = list(set(range(i)) - set(remove))
        new_x = x[keep]

        self._memories = torch.as_tensor(new_x, device=device(), dtype=torch.float32)
