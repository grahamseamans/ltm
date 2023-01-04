import torch
import numpy as np


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def freaky_take_3d_2nd(src, idxs, dim):
    assert len(idxs) == src.shape[dim]
    a, b, c = src.shape
    ret = torch.ones(a, c).to(device())
    for i, idx in zip(range(a), idxs):
        ret[i, :] = src[i, idx, :]
    return ret


if __name__ == "__main__":
    acts = torch.rand((3, 4, 6))  # 3 4 6
    rets = torch.rand((3, 4, 2))
    rets = torch.einsum("ijk->ij", rets)  # 3 4
    maxes = torch.argmax(rets, axis=1)  # 3
    selector = torch.zeros((3, 4)).bool()
    print(selector)
    # selector[:] = False
    for i, x in enumerate(maxes):
        selector[i, x] = True
    _acts = acts[selector, :]
    print(acts)
    print(_acts.shape)
    print(_acts)
    print("b")
    # print(__acts.shape)
    # print(__acts)
    # acts = mems[..., : self.memory.act_len]  # 10 16 6
    # rets = mems[..., self.memory.act_len :]  # 10 16 2
    # rets = torch.einsum("ijk->ij", rets)  # 10 16
    # maxes = torch.argmax(rets, dim=1)  # 10
    # acts = acts[:, maxes, :] # 10 10 6
    # class a:
    #     def __init__(self, a) -> None:
    #         self.mem = a
    # mem = a(4)
    # t_1 = a(mem)
    # t_2 = a(mem)

    # print(t_1.mem.mem)
    # print(t_2.mem.mem)

    # mem.mem = 5

    # print(t_1.mem.mem)
    # print(t_2.mem.mem)

    # mem = t_1.mem
    # mem.mem = 6

    # print(t_1.mem.mem)
    # print(t_2.mem.mem)
