import torch


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":

    class a:
        def __init__(self, a) -> None:
            self.mem = a

    mem = a(4)
    t_1 = a(mem)
    t_2 = a(mem)

    print(t_1.mem.mem)
    print(t_2.mem.mem)

    mem.mem = 5

    print(t_1.mem.mem)
    print(t_2.mem.mem)

    mem = t_1.mem
    mem.mem = 6

    print(t_1.mem.mem)
    print(t_2.mem.mem)
