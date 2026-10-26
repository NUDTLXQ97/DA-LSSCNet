"""Small, explicit helpers for single-node PyTorch DDP training.

The project uses torchrun environment variables when available and otherwise
falls back to ordinary single-process execution.  Training scripts keep all
checkpointing, validation, and console output on rank 0.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableSequence, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(requested_device: str = "cuda") -> DistributedContext:
    """Initialize torch.distributed from torchrun variables when present."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    wants_cuda = requested_device.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    if wants_cuda:
        if distributed:
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible."
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            # Respect an explicit cuda:N; otherwise use the first visible GPU.
            device = torch.device(requested_device if ":" in requested_device else "cuda:0")
            torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        init_kwargs = {"backend": backend, "init_method": "env://"}
        # Newer PyTorch releases accept device_id and can eagerly establish the
        # NCCL communicator.  Passing it also avoids ambiguous rank-to-GPU
        # warnings.  The fallback keeps compatibility with older releases.
        if device.type == "cuda":
            try:
                dist.init_process_group(**init_kwargs, device_id=device)
            except TypeError:
                dist.init_process_group(**init_kwargs)
        else:
            dist.init_process_group(**init_kwargs)
        _safe_barrier()

    return DistributedContext(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def _safe_barrier() -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        _safe_barrier()
        dist.destroy_process_group()


def barrier() -> None:
    _safe_barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(
    model: torch.nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if not context.distributed:
        return model
    if context.device.type == "cuda":
        return DDP(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,
        )
    return DDP(
        model,
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
    )


def rank_print(context: DistributedContext, *args: Any, **kwargs: Any) -> None:
    if context.is_main:
        print(*args, **kwargs)


def set_rank_seed(base_seed: int, context: DistributedContext) -> int:
    seed = int(base_seed) + int(context.rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def seed_worker(worker_id: int) -> None:
    # torch.initial_seed() is already unique per process and worker.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def broadcast_object(value: Any, context: DistributedContext, src: int = 0) -> Any:
    if not context.distributed:
        return value
    holder: MutableSequence[Any] = [value if context.rank == src else None]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def all_reduce_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def reduce_scalar_dict(
    values: Mapping[str, float],
    context: DistributedContext,
    *,
    average: bool = False,
) -> Dict[str, float]:
    keys = list(values.keys())
    tensor = torch.tensor(
        [float(values[key]) for key in keys],
        dtype=torch.float64,
        device=context.device,
    )
    all_reduce_sum(tensor, context)
    if average and context.world_size > 1:
        tensor /= context.world_size
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


class DistributedWeightedSampler(Sampler[int]):
    """DDP-aware weighted sampling with a *global* samples-per-epoch budget.

    A single deterministic multinomial draw of ``total_size`` indices is
    conceptually made for the whole job, then each rank receives a strided
    shard. This avoids every rank independently drawing the full epoch.
    """

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        global_num_samples: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        replacement: bool = True,
        seed: int = 0,
    ) -> None:
        if global_num_samples <= 0:
            raise ValueError("global_num_samples must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank={rank} is invalid for world_size={world_size}")
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        if self.weights.numel() == 0 or not torch.isfinite(self.weights).all():
            raise ValueError("Sampling weights must be finite and non-empty")
        if float(self.weights.sum()) <= 0:
            raise ValueError("At least one sampling weight must be positive")
        self.global_num_samples = int(global_num_samples)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(self.global_num_samples / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        indices = indices[self.rank : self.total_size : self.world_size]
        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"Sampler shard has {len(indices)} indices, expected {self.num_samples}"
            )
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def distributed_sampler_from_weighted(
    weighted_sampler: Any,
    global_num_samples: int,
    context: DistributedContext,
    seed: int,
) -> DistributedWeightedSampler:
    weights = getattr(weighted_sampler, "weights", None)
    if weights is None:
        raise TypeError("Expected a WeightedRandomSampler-like object with .weights")
    return DistributedWeightedSampler(
        weights=weights,
        global_num_samples=global_num_samples,
        rank=context.rank,
        world_size=context.world_size,
        replacement=True,
        seed=seed,
    )

class DistributedEvalSampler(Sampler[int]):
    """Shard a validation dataset without padding or duplicate samples."""

    def __init__(self, dataset_size: int, rank: int = 0, world_size: int = 1) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.dataset_size:
            return 0
        return int(math.ceil((self.dataset_size - self.rank) / self.world_size))
