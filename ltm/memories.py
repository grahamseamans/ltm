import torch
import torch.nn.functional as f
import random
import math
from utils import device
from tianshou.data import Batch
import einops
import numpy as np
import scipy
from sklearn.preprocessing import normalize, MinMaxScaler
import time


class DangerMemories:
    def __init__(self, mem_thresh):
        self.mem_thresh = mem_thresh
        self.danger_table = None
        self.device = device()

    def add(self, batch: Batch):
        danger_obs = batch.obs[batch.terminated]
        if not isinstance(self.danger_table, np.ndarray):
            self.danger_table = np.copy(danger_obs)
        else:
            self.danger_table = np.concatenate((self.danger_table, danger_obs))

        self.dream()
        self._torch_danger_table = (
            torch.from_numpy(self.danger_table).to(self.device).float()
        )
        self._torch_danger_table.requires_grad = False

    def dream(self):
        mems = self.danger_table
        if len(mems) <= self.mem_thresh:
            return

        tree = scipy.spatial.KDTree(mems)
        close = tree.query(mems, k=(len(mems) - self.mem_thresh + 1))[0]
        close = close[:, 1:]

        sorted = np.argsort(close.flatten())
        ordered = np.unravel_index(sorted, close.shape)[0]
        pairs = ordered.reshape((len(ordered) // 2, 2))

        remove = []
        idx = 0
        while len(remove) < len(mems) - self.mem_thresh:
            a, b = pairs[idx]
            idx += 1
            if a not in remove and b not in remove:
                remove.append(a if random.random() > 0.5 else b)

        keep = list(set(range(len(mems))) - set(remove))
        self.danger_table = mems[keep]


class Memories:
    def __init__(self, obs_len, rew_len, act_len, mem_thresh):
        self.mem_thresh = mem_thresh

        self.ret_len = 2
        self.obs_len = math.prod(obs_len)
        self.act_len = math.prod(act_len)
        self.mem_len = self.obs_len + self.ret_len + self.act_len

        self._np_memories = np.ones((0, self.mem_len))
        self._memories = torch.ones(0, self.mem_len).to(device()).float()
        self._dummy = torch.rand(self.mem_thresh, self.mem_len).to(device()).float()
        self._dummy.requires_grad = False
        self.mem_empty = True
        self.time_table = None
        self.time_step = 0

    @property
    def memories(self):
        return self._dummy if self.mem_empty else self._memories

    def add(self, batch: Batch):
        self.time_step += 1
        mems_needed = self.mem_thresh // 4
        if self.mem_empty:
            mems_needed = self.mem_thresh - len(self._memories)
            if len(self._memories) + mems_needed >= self.mem_thresh:
                self.mem_empty = False

        if not isinstance(self.time_table, np.ndarray):
            self.time_table = np.ones(mems_needed)
            self.time_table[:] = self.time_step
        else:
            new_times = np.ones(mems_needed)
            new_times[:] = self.time_step
            self.time_table = np.concatenate((self.time_table, new_times))

        to_add = next(batch.split(mems_needed, shuffle=True))
        to_add = self.pre_process_mem_bits(to_add, modify_data=False)
        new_memories = np.concatenate((to_add.obs, to_add.act, to_add.returns), axis=1)
        self._np_memories = np.concatenate((self._np_memories, new_memories))

        self.dream()
        self._memories = torch.from_numpy(self._np_memories).to(device()).float()
        self._memories.requires_grad = False
        # newest_mem = np.max(self.time_table)
        # oldtest_mem = np.min(self.time_table)
        # avg_mem_age = np.mean(self.time_table)
        # print(
        #     f"current age: {self.time_step}, avg_mem_age: {avg_mem_age} newest_mem: {newest_mem} oldest_mem: {oldtest_mem}"
        # )
        # unique, counts = np.unique(self.time_table, return_counts=True)
        # print(dict(zip(unique, counts)))

    def pre_process_mem_bits(self, batch: Batch, modify_data=False):
        if not modify_data:
            batch = Batch(batch, copy=True)

        if "returns" in batch:
            if isinstance(batch.returns, torch.Tensor):
                batch.returns = batch.returns.cpu().numpy()
            batch.returns = einops.repeat(batch.returns, "m -> m k", k=2)
            batch.returns[:, 1] = 1
            batch.returns = normalize(batch.returns)

        if isinstance(batch.act, torch.Tensor):
            batch.act = batch.act.cpu().numpy()

        batch.obs = normalize(batch.obs)
        batch.act = normalize(batch.act)

        return batch

    def add_boredom(self, batch: Batch):
        if self.mem_empty:
            return batch

        # get the bits
        normed = self.pre_process_mem_bits(batch)
        new_experiences = np.concatenate((normed.obs, normed.act), axis=1)
        mems = self._np_memories[:, : self.obs_len + self.act_len]

        # normalize for len 1 vects
        mems = normalize(mems)
        new_experiences = normalize(new_experiences)

        # get tree
        tree = scipy.spatial.KDTree(mems)

        # boredom
        query_ret = tree.query(new_experiences, k=1)
        close = query_ret[0]
        close = np.clip(close, 0.01, 0.99)
        batch.rew *= np.abs(close - 1)

        # nostalgia
        query_ret = tree.query(new_experiences, k=10)
        idxs = query_ret[1]
        for i, idx in enumerate(idxs):
            nostalgia = np.abs(np.mean((self.time_table[idx] / self.time_step)) - 1)
            batch.rew[i] += nostalgia

        return batch

    def print_stats(thing: np.ndarray, name: str = None):
        print(name, np.max(thing), np.min(thing), np.mean(thing))

    def dream(self):
        mems = self._np_memories
        if len(mems) <= self.mem_thresh:
            return
        num_input_mems = len(mems)
        i_thresh = self.mem_thresh
        mems_no_ret = mems[:, : self.obs_len + self.act_len]

        tree = scipy.spatial.KDTree(mems_no_ret)
        close = tree.query(mems_no_ret, k=(num_input_mems - i_thresh + 1))[0]
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
                remove.append(a if self.time_table[a] < self.time_table[b] else b)

        keep = list(set(range(num_input_mems)) - set(remove))

        self._np_memories = mems[keep]
        self.time_table = self.time_table[keep]
