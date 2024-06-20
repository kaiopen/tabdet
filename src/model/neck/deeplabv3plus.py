from typing import Any, Dict, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.nn.conv import Conv2dBlock
from kaitorch.zoo.deeplabv3plus import ASPP


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (128, 2048),
        dilations: Sequence[int] = (6, 12, 18),
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._project = Conv2dBlock(
            in_channels[0], 48, 1,
            activation=activation,
            activation_kw=activation_kw
        )
        self._aspp = ASPP(
            in_channels[1], dilations, padding_mode, activation, activation_kw
        )

    def forward(
        self, x: Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]
    ) -> TorchTensor[TorchFloat]:
        x_0, x_1 = x
        return torch.cat(
            (
                self._project(x_0),
                F.interpolate(
                    self._aspp(x_1), size=x_0.shape[2:], mode='bilinear'
                )
            ),
            dim=1
        )
