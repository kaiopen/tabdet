from typing import Any, Dict, Sequence, Tuple, Union
from pathlib import Path
import random

import torch

from kaitorch.typing import TorchTensor, TorchBool, TorchFloat, TorchInt64, \
    real
from kaitorch.pcd import PointCloudXYZIR
from tab import TAB

from ..dataset import Dataset as _Dataset
from ..bev import BEVEncoder


class Dataset(_Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        split: str = 'train',
        is_training: bool = False,
        size: Tuple[real, real] = (256, 512),
        num_neighbor: int = 0,
        dir: Union[Path, str] = Path.cwd().joinpath('tmp', 'TAB'),
        *args, **kwargs
    ) -> None:
        r'''Interface class of a dataset for training.

        ### Args:
            - root: path to the TAB dataset.
            - split: "train", "val" or "test".
            - is_training: Whether to call for training.

            - size: size of a BEV map. The form of it should be `(H, W)`.
            - num_neighbor: number of neighbors to be sampled.
            - dir: path to the preprocessed information.

        ### Methods:
            - __getitem__
            - __iter__
            - __len__
            - __next__

        ### Static Methods:
            - cuda

        __getitem__
        ## Args:
            - index

        ### Returns:
            - Information about the frame.
            - Tuple from a BEV encoder.
            - Additional information from preprocess if the dataset is called
                for training or `None`.

        '''
        super().__init__()
        self._tab = TAB(root, split)

        self._encode = BEVEncoder(
            range_xy=list(TAB.RANGE_X) + list(TAB.RANGE_Y),
            ground=TAB.GROUND,
            num_ring=TAB.NUM_RING,
            size=size,
            num_neighbor=num_neighbor
        )

        if is_training:
            if isinstance(dir, str):
                self._dir = Path(dir)
            else:
                self._dir = dir
            self._get = self._getitem_for_training
        else:
            self._get = self._getitem_for_inference

        self.__i = 0
        self._len = len(self._tab)

    def __getitem__(
        self, index: int
    ) -> Tuple[
        TAB.Frame,
        Tuple[
            TorchTensor[TorchFloat],
            TorchTensor[TorchInt64],
            TorchTensor[TorchBool]
        ],
        Union[Dict[str, Any], None]
    ]:
        f, pcd, info = self._get(index)

        # Shuffle points.
        indices = [i for i in range(len(pcd))]
        random.shuffle(indices)
        pcd.filter_(indices)

        return f, self._encode(pcd), info

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self._len

    def __next__(self):
        if (i := self.__i) < self._len:
            data = self[i]
            self.__i += 1
            return data
        self.__i = 0
        raise StopIteration

    def _getitem_for_training(
        self, index: int
    ) -> Tuple[
        TAB.Frame,
        PointCloudXYZIR,
        Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]
    ]:
        f = self._tab[index]
        pcd = self._tab.get_pcd(f)
        targets = torch.load(self._dir.joinpath(f.sequence, f.id + '.pth'))

        # Flip randomly.
        if random.random() > 0.5:
            pcd.flip_around_x_axis_()

            hm, off = targets
            targets = (torch.flip(hm, dims=(2,)), torch.flip(off, dims=(2,)))

        return f, pcd, targets

    def _getitem_for_inference(
        self, index: int
    ) -> Tuple[TAB.Frame, PointCloudXYZIR, None]:
        f = self._tab[index]
        return f, self._tab.get_pcd(f), None

    @staticmethod
    def collate(
        x: Sequence[
            Tuple[
                TAB.Frame,
                Tuple[
                    TorchTensor[TorchFloat],
                    TorchTensor[TorchInt64],
                    TorchTensor[TorchBool]
                ],
                Union[
                    Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]],
                    None
                ]
            ]
        ]
    ) -> Tuple[
        Sequence[TAB.Frame],
        Tuple[
            TorchTensor[TorchFloat],
            TorchTensor[TorchInt64],
            TorchTensor[TorchBool]
        ],
        Union[
            Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]], None
        ]
    ]:
        frames = []

        nebs = []
        inds = []
        masks = []

        hms = []
        offs = []

        num = 0
        for f, neb, t in x:
            frames.append(f)

            # (Xi, C), (L, num_neighbor), (1, HW)
            neb, ind, mask = neb
            ind[ind != -1] += num
            num += len(neb)

            nebs.append(neb)
            inds.append(ind)
            masks.append(mask)

            if t is not None:
                # (1, H, W, num_cat), (1, H, W, 2)
                hm, off = t
                hms.append(hm)
                offs.append(off)

        if t is not None and len(hms) != len(x):
            raise ValueError('invalid batch size of additional information.')

        return frames, (
            torch.cat(nebs, dim=0),  # (X = X0 + X1 + ..., D)
            torch.cat(inds, dim=0),  # (L = sum(mask), k)
            torch.cat(masks, dim=0),  # (B, M)
        ), (
            torch.cat(hms, dim=0),  # (B, H, W, num_cat)
            torch.cat(offs, dim=0)  # (B, H, W, 2)
        ) if t is not None else None

    @staticmethod
    def cuda(
        x: Tuple[
            Sequence[TAB.Frame],
            Tuple[
                TorchTensor[TorchFloat],
                TorchTensor[TorchInt64],
                TorchTensor[TorchBool]
            ],
            Union[
                Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]], None
            ]
        ]
    ) -> Tuple[
        Sequence[TAB.Frame],
        Tuple[
            TorchTensor[TorchFloat],
            TorchTensor[TorchInt64],
            TorchTensor[TorchBool]
        ],
        Union[
            Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]], None
        ]
    ]:
        r'''Convert tensors onto CUDA.

        ### Args:
            x: a sequence of tensors.

        ### Returns:
            - Tensors on CUDA.

        '''
        f, x, t = x
        return f, [v.cuda(non_blocking=True) for v in x], \
            None if t is None else [v.cuda(non_blocking=True) for v in t]
