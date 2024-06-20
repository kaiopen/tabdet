from typing import Any, Optional
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    name: str,
    epoch: int,
    model: Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None
) -> str:
    root = Path.cwd().joinpath('checkpoints', name)
    root.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()

    p = root.joinpath(f'{epoch:0>3d}.pth')
    torch.save(state, p)
    return str(p)


def load_checkpoint_(
    name: str,
    model: Module,
    checkpoint: Optional[str] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    map_location: Any = 'cpu'
) -> int:
    root = Path.cwd().joinpath('checkpoints', name)
    if checkpoint is None:
        c = torch.load(max(root.glob('*.pth')), map_location=map_location)
    else:
        c = torch.load(
            root.joinpath(checkpoint + '.pth'), map_location=map_location
        )

    model.load_state_dict(c['model'])
    if optimizer is not None:
        optimizer.load_state_dict(c['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(c['scheduler'])
    if scaler is not None:
        scaler.load_state_dict(c['scaler'])
    return c['epoch']
