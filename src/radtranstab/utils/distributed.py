from __future__ import annotations

import os 
import torch
import torch.distributed as dist 


def setup_distributed(distributed: bool):
    """setup distributed setting"""
    if not distributed:
        return 0, 0, 1, "cuda:0" if torch.cuda.is_available() else "cpu"

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, f"cuda:{local_rank}"


def cleanup_distributed(distributed: bool):
    """cleanup distributed setting"""
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """check whether currently executing process is main process"""
    return not dist.is_initialized() or dist.get_rank() == 0

