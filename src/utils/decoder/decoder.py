from typing import Any, Dict, Sequence, Tuple

import torch

from kaitorch.typing import TorchTensor, TorchTensorLike, \
    TorchDevice, TorchInt64, TorchFloat, Real, real
from kaitorch.data import ReverseGroup, cell_from_size

from src.data import NMS
from ..cluster import CLUSTER


class Decoder:
    def __init__(
        self,
        range_xy: TorchTensorLike[Real],
        categories: Sequence[str],
        nms: Dict[str, Any],
        cluster: Dict[str, Any],

        size: TorchTensorLike[TorchInt64] = (256, 512),
        scale: int = 1,
        threshold: float = 0.2,
        device: TorchDevice = torch.device('cpu'),
        *args, **kwargs
    ) -> None:
        r'''

        ### Args:
            - coord: 'xy' or 'rt'. 'xy' for the rectangular coordinate system.
                And 'rt' for the polar coordinate system.
            - range_xy
            - categories

            - size: size of the BEV map. The form of it should be `(H, W)`.
            - cell: size of a cell. The form of it should be `(H, W)`.
            - error: tolerant error. The form of it should be `(H, W)`.

            - scales: scales of the model.
            - threshold: score threshold.
            - radius: cluster radius.
            - device

        ### Methods:
            - __call__

        __call__
        ### Args:
            - x: a sequence of predicted results.

        ### Returns:
            - A sequence of predicted boundaries of some frames.

        '''
        self._categories = categories
        self._scale = scale
        self._t = threshold
        self._nms = NMS[nms.pop('type')](device=device, **nms)
        self._cluster = CLUSTER[cluster.pop('type')](**cluster)

        range_xy = torch.as_tensor(range_xy, device=device)
        lower_bound = range_xy[[0, 2]]
        self._reverse = ReverseGroup(
            lower_bound,
            cell_from_size(
                lower_bound, range_xy[[1, 3]],
                torch.as_tensor(size, device=device),
            ),
            device=device
        )
        self._device = device

    @torch.no_grad()
    def __call__(
        self, x: Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]
    ) -> Sequence[Sequence[Tuple[str, Sequence[Tuple[real, real]]]]]:
        scores, points = x
        scores = scores.to(self._device)
        points = points.to(self._device)

        mask = scores.sigmoid_() >= self._t
        # (X,) X = sum(mask)
        bats, cats, inds_h, inds_w = torch.nonzero(mask, as_tuple=True)
        scores = scores[mask]  # (X,)
        points = points.permute(0, 2, 3, 1)[bats, inds_h, inds_w]  # (X, 2)
        points[..., 0] += inds_h
        points[..., 1] += inds_w
        points = self._reverse(points * self._scale)

        frames = []
        for b in range(mask.shape[0]):
            m = bats == b
            _cats = cats[m]
            _scores = scores[m]
            _points = points[m]

            m = self._nms(_scores, _points)
            _cats = _cats[m]
            _points = _points[m]

            frame = []
            for c, cat in enumerate(self._categories):
                m = _cats == c
                if not torch.any(m):
                    continue
                pts = _points[m]

                clusters = self._cluster(pts)
                for i in range(max(clusters) + 1):
                    frame.append((cat, pts[clusters == i].cpu().tolist()))
            frames.append(frame)
        return frames
