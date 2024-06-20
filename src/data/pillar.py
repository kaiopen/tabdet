from typing import Sequence, Tuple

import torch

from kaitorch.typing import TorchTensor, \
    TorchBool, TorchFloat, TorchInt64, real


class Pillar:
    def __init__(
        self, num_pillar: int, num_neighbor: int, z: real,
        *args, **kwargs
    ) -> None:
        r'''Extract neighbors for each pillar.

        ### Args:
            - num_pillar: number of pillars.
            - num_neighbor: number of neighbors to be sampled for each pillar.
                It should be larger than 0.
            - z: center z of a pillar.

        ### Methods:
            - __call__

        __call__
        ### Args:
            - points_p: 3D coordinates of some points. Its shape should be
                `(N, 3)`.
            - points_f: features of the points. Its shape should be `(N, C)`.
            - groups: groups of the points. Its shape should be `(N, 2)`.
            - ids: IDs of nonempty pillars. Its length should be `M`.
            - indices: indices of the points in each nonempty pillar. Its
                length should be `M`.

        ### Returns:
            - Neighbors. Its shape is `(X, 6 + C)`.
            - Indices for picked neighbors. Its shape is
                `(L < H * W, num_neighbor)`
            - Mask indexing nonempty pillars. Its shape is `(1, H * W)`. And it
                meets `sum(mask) == L`.

        '''
        if num_neighbor <= 0:
            raise ValueError(
                f'an invalid number of neighbors ({num_neighbor}).'
            )

        self._num_pillar = num_pillar
        self._k = num_neighbor
        self._z = z

    def __call__(
        self,
        points_p: TorchTensor[TorchFloat],
        points_f: TorchTensor[TorchFloat],
        groups: TorchTensor[TorchInt64],
        ids: Sequence[int],
        indices: Sequence[TorchTensor[TorchInt64]],
        *args, **kwargs
    ) -> Tuple[
        TorchTensor[TorchFloat],
        TorchTensor[TorchInt64],
        TorchTensor[TorchBool]
    ]:
        inds = []
        means = []
        mask = torch.zeros((self._num_pillar, self._k), dtype=bool)
        for id, _indices in zip(ids, indices):
            # Pick out the first k neighbor points in the pillar.
            _inds = _indices[:self._k]
            inds.append(_inds)
            num = len(_inds)  # <= k

            # The centroid of a pillar.
            means.append(
                torch.unsqueeze(
                    torch.mean(points_p[_indices], dim=0), dim=0
                ).expand(num, 3)
            )
            mask[id, :num] = True

        # Total X points are picked out. X <= M * k
        inds = torch.cat(inds)  # (X,)
        points_p = points_p[inds]

        m = torch.any(mask, dim=-1)  # whether a pillar is nonempty
        # (L = sum(m) <= num_pillar, k)
        indices = -torch.ones((torch.sum(m), self._k), dtype=torch.long)
        # For nonempty pillars, the indices of their neighbors.
        indices[mask[m]] = torch.arange(len(inds))
        return torch.cat(
            (
                points_p - torch.cat(means, dim=0),  # centroid
                points_p[:, :2] - groups[inds] - 0.5,
                points_p[:, 2: 3] - self._z,  # center
                points_f[inds]
            ),
            dim=-1
        ), indices, m.unsqueeze_(0)
