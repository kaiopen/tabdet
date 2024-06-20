from typing import Optional
import json
import os

from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from kaitorch.utils import Configer, Logger

from src.data import DATASET
from src.model import Model
from .criterion import CRITERION
from .checkpoint import load_checkpoint_, save_checkpoint
from .utils import process_logs


OPTIMIZER = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW
}

SCHEDULER = {
    'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingLR
}


class DDPTrainer:
    def __init__(self, cfg: Configer, *args, **kwargs) -> None:
        # NOTE: Keep this call or an open-many-files error occurs.
        torch.multiprocessing.set_sharing_strategy('file_system')

        local_rank = int(os.environ['LOCAL_RANK'])
        self._is_rank_0 = local_rank in (-1, 0)
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )

        self._name = cfg.name
        self._start_epoch = 0
        self._end_epoch = cfg.run.end_epoch

        # LOGGER
        self._logger = Logger(
            'train_ddp_' + self._name, level=Logger.INFO
        ).info if local_rank in (-1, 0) else lambda *args, **kwargs: None
        self._logger('\n' + str(cfg))

        # DATALOADER
        params = cfg.dataset.dict()
        self._dataset = DATASET[params.pop('type')](**params)
        self._sampler = DistributedSampler(self._dataset)
        self._loader = DataLoader(
            self._dataset,
            batch_size=cfg.run.batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=cfg.run.num_worker,
            collate_fn=self._dataset.collate,
            pin_memory=True,
            drop_last=True
        )
        self._num_batch = len(self._loader)
        self._num_acc = max(round(64 / cfg.run.batch_size), 1)
        self._logger('=== The DATALOADER has been READY! ===')

        # MODEL
        self._model = Model(**cfg.model.dict())
        self._model = self._model.cuda()
        self._m = DistributedDataParallel(
            SyncBatchNorm.convert_sync_batchnorm(self._model),
            device_ids=[local_rank]
        )
        self._logger('=== The MODEL has been READY! ===')

        # CRITERION
        params = cfg.criterion.dict()
        self._criterion = CRITERION[params.pop('type')](**params)
        self._logger('=== The CRITERION has been READY! ===')

        # OPTIMIZER
        params = cfg.optimizer.dict()
        self._optimizer = OPTIMIZER[params.pop('type')](
            self._m.parameters(), **params
        )
        self._logger('=== The OPTIMIZER has been READY! ===')

        # SCHEDULER
        params = cfg.scheduler.dict()
        self._scheduler = SCHEDULER[params.pop('type')](
            self._optimizer, **params
        )
        self._logger('=== The SCHEDULER has been READY! ===')

        self._amp = cfg.run.amp
        if self._amp:
            self._scaler = GradScaler()
            self._logger('=== The AMP has been READY! ===')

        if cfg.run.resume:
            self._logger(
                f'=== RESUMED checkpoint EPOCH {
                    self.resume(cfg.run.checkpoint)
                } ==='
            )

    def __call__(self):
        self._logger('\n=== TRAIN ===')
        self._m.train()
        for epoch in range(self._start_epoch, self._end_epoch):
            self._sampler.set_epoch(epoch)
            self.train_epoch(epoch)
        self._logger('\n=== DONE ===')

    def resume(self, checkpoint: Optional[str] = None):
        epoch = load_checkpoint_(
            name=self._name,
            model=self._model,
            checkpoint=checkpoint,
            optimizer=self._optimizer,
            scheduler=self._scheduler
        )
        self._start_epoch = epoch + 1
        return epoch

    def save_checkpoint(self, epoch: int) -> str:
        return save_checkpoint(
            name=self._name,
            epoch=epoch,
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler
        )

    def train_epoch(self, epoch: int):
        loader = self._loader
        if self._is_rank_0:
            loader = tqdm(loader, desc=f'EPOCH {epoch:0>3d}')
            loader.set_postfix_str('loss=0.00000')
            logs = []

        self._optimizer.zero_grad()
        for it, x in enumerate(loader):
            _, x, targets = self._dataset.cuda(x)
            if self._amp:
                with autocast():
                    loss, log = self._criterion(self._m(x), targets)
                self._scaler.scale(loss).backward()

                if 0 == (it + epoch * self._num_batch) % self._num_acc:
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad()
            else:
                loss, log = self._criterion(self._m(x), targets)
                loss.backward()
                if 0 == (it + epoch * self._num_batch) % self._num_acc:
                    self._optimizer.step()
                    self._optimizer.zero_grad()
            if self._is_rank_0:
                loader.set_postfix_str(f'loss={loss.item():.5f}')
                logs.append(log)
            # torch.cuda.empty_cache()
        self._scheduler.step()

        if self._is_rank_0:
            self.save_checkpoint(epoch)
            self._logger(
                f'EPOCH {epoch}:\n'
                + json.dumps(process_logs(logs), indent=2),
                False
            )
