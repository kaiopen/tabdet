from typing import Tuple
import torch

from kaitorch.typing import TorchTensor, TorchDevice, TorchFloat, TorchInt64
from kaitorch.data import Group, cell_from_size, mask_in_range, xy_to_rt
from tab import TAB

from ..nms import NMS as _NMS


class NMS(_NMS):
    def __init__(
        self,
        size: Tuple[int, int] = (64, 128),
        device: TorchDevice = torch.device('cpu'),
        *args, **kwargs
    ) -> None:
        r'''


        '''
        super().__init__(*args, **kwargs)
        self._r = list(TAB.RANGE_RHO) + list(TAB.RANGE_THETA)
        self._w = size[1]

        lower_bound = torch.as_tensor((TAB.RANGE_X[0], TAB.RANGE_Y[0]))
        upper_bound = torch.as_tensor((TAB.RANGE_X[1], TAB.RANGE_Y[1]))
        self._group = Group(
            lower_bound,
            cell=cell_from_size(
                lower_bound, upper_bound, torch.as_tensor(size)
            ),
            upper_bound=upper_bound,
            device=device
        )

    def __call__(
        self, scores: TorchTensor[TorchFloat], points: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchInt64]:
        mask = mask_in_range(xy_to_rt(points), self._r)
        scores = scores[mask]
        points = points[mask]

        groups = self._group(points)
        ids, indices = torch.sort(groups[:, 0] * self._w + groups[:, 1])
        _, counts = torch.unique_consecutive(ids, return_counts=True)

        out = []
        i = 0
        for c in torch.cumsum(counts, dim=0).tolist():
            inds = indices[i: c]
            out.append(inds[torch.argmax(scores[inds], dim=0)])
            i = c
        return torch.nonzero(
            mask, as_tuple=True
        )[0][torch.as_tensor(out, device=scores.device)]
