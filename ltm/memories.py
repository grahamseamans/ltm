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
import time


class Memories:
    def __init__(self, obs_len, rew_len, act_len, mem_thresh):
        self.mem_thresh = mem_thresh

        self.rew_len = 2
        self.obs_len = math.prod(obs_len)
        self.act_len = math.prod(act_len)
        total_len = self.obs_len + self.rew_len + self.act_len

        self._np_memories = np.ones((0, total_len))
        self._memories = torch.ones(0, total_len).to(device()).float()
        self._dummy = torch.rand(self.mem_thresh, total_len).to(device()).float()
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

        rew = rew.cpu().numpy()
        act = act.cpu().numpy()

        obs = normalize(obs)
        rew = normalize(rew)
        act = normalize(act)

        new_memories = np.concatenate((obs, rew, act), axis=1)
        self._np_memories = np.concatenate((self._np_memories, new_memories))

        self.dream()
        self._memories = torch.from_numpy(self._np_memories).to(device()).float()

    def dream(self):
        mems = self._np_memories
        if len(mems) <= self.mem_thresh:
            return
        num_input_mems = len(mems)
        i_thresh = self.mem_thresh

        tree = scipy.spatial.KDTree(mems)
        query_ret = tree.query(mems, k=(num_input_mems - i_thresh + 1))
        close = query_ret[0]
        close = close[:, 1:]

        sorted = np.argsort(close.flatten())
        ordered = np.unravel_index(sorted, close.shape)[0]
        pairs = ordered.reshape((len(ordered) // 2, 2))

        remove = []
        idx = 0
        while len(remove) < num_input_mems - i_thresh:
            a, b = pairs[idx]
            idx += 1
            if a not in remove and b not in remove:
                remove.append(a if random.random() > 0.5 else b)

        keep = list(set(range(num_input_mems)) - set(remove))

        self._np_memories = mems[keep]
