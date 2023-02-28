import torch.distributed as dist
import torch
from torch._C._distributed_c10d import ReduceOp


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_rank():
    if not is_distributed():
        return 0

    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def all_reduce_mean(tensor: torch.Tensor):
    if is_distributed():
        dist.all_reduce(tensor, op=ReduceOp.SUM)
        return tensor / get_world_size()
    else:
        return tensor

