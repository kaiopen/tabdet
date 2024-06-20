from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.zoo.focal_loss import focal_loss


def clamp_sigmoid(x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)


class Criterion:
    def __init__(self, *args, **kwargs) -> None:
        r'''

        ### Methods:
            - __call__

        __call__
        ### Args:
            - x: A heatmap with shape `(B, num_category, H, W)` and an offset
                map with shape `(B, 2, H, W)`.
            - targets: a ground truth heatmap with shape
                `(B, H, W, num_category)` and a ground truth offset map with
                shape `(B, H, W, 2)`.

        ### Returns:
            - Loss.
            - Log.

        '''
        return

    def __call__(
        self,
        x: Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]],
        targets: Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]
    ) -> Tuple[TorchFloat, Dict[str, Any]]:
        hm_x, off_x = x
        hm_x = hm_x.permute(0, 2, 3, 1)  # (B, H, W, num_category)
        off_x = off_x.permute(0, 2, 3, 1)  # (B, H, W, 2)

        hm_t, off_t = targets
        mask = torch.any(1 == hm_t, dim=-1)

        loss_cls = focal_loss(clamp_sigmoid(hm_x), hm_t)
        loss_off = F.smooth_l1_loss(off_x[mask], off_t[mask])
        loss = loss_cls + 5 * loss_off
        return loss, {
            'cls': loss_cls.item(),
            'off': loss_off.item(),
            'loss': loss.item()
        }
