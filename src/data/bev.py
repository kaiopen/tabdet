from typing import Tuple

import torch

from kaitorch.typing import TorchTensor, TorchTensorLike, \
    TorchBool, TorchFloat, TorchInt64, Real, real
from kaitorch.data import Group, cell_from_size
from kaitorch.pcd import PointCloudIRs

from .pillar import Pillar


class BEVEncoder:
    def __init__(
        self,
        range_xy: TorchTensorLike[Real],
        ground: real,
        size: Tuple[real, real] = (256, 512),
        num_neighbor: int = 20,
        *args, **kwargs
    ) -> None:
        r'''

        ### Args:
            - range_xy
            - ground: height of ground.
            - num_ring
            - size: size of a BEV map. The form of it should be `(H, W)`.
            - group: group function.
            - num_neighbor: number of neighbors to be sampled.

        ### Methods:
            - __call__

        __call__
        ### Args:
            - pcd

        ### Returns:
            - `None` if the `num_neighbor` is a zero else a tuple including
                - Neighbors. Its shape is `(X, C)`.
                - Indices for picked neighbors. Its shape is
                    `(L < H * W, num_neighbor)`.
                - Mask indexing nonempty grid. Its shape is `(1, H * W)`. And
                    it meets `sum(mask) == L`.
            - Artifically designed BEV. Its shape is `(1, F, H, W)`. Or a
                `None` if no projector is called.

        '''

        range_xy = torch.as_tensor(range_xy)
        lower_bound = range_xy[[0, 2]]
        upper_bound = range_xy[[1, 3]]
        self._group = Group(
            lower_bound,
            cell=cell_from_size(
                lower_bound, upper_bound, torch.as_tensor(size)
            ),
            upper_bound=upper_bound
        )

        h, self._w = size
        self._pillar = Pillar(h * self._w, num_neighbor, ground)

    def __call__(
        self, pcd: PointCloudIRs
    ) -> Tuple[
        TorchTensor[TorchFloat],
        TorchTensor[TorchInt64],
        TorchTensor[TorchBool]
    ]:
        points_p = pcd.xyz_.float()

        # Pick up points in each pillar.
        groups = self._group(points_p[:, :2])  # (N, 2)
        # Assign a grid ID to each point.
        # Sort points by IDs.
        ids, indices = torch.sort(groups[:, 0] * self._w + groups[:, 1])
        # Count the number of points included in a grid.
        ids, counts = torch.unique_consecutive(ids, return_counts=True)

        inds = []
        i = 0
        for c in torch.cumsum(counts, dim=0).tolist():
            inds.append(indices[i: c])
            i = c

        return self._pillar(
            points_p=points_p,
            points_f=torch.cat(
                (points_p, pcd.rt_, pcd.ring_.float(), pcd.intensity_.float()),
                dim=-1
            ),
            groups=groups,
            ids=ids.tolist(),
            indices=inds
        )
