from typing import Any, Dict, Sequence, Union

from torch import nn

from kaitorch.nn.conv import Conv2dBlock
from kaitorch.zoo.hrnet import HRNet as _HRNet


class HRNet(_HRNet):
    def __init__(
        self,
        in_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__(
            in_channels, padding_mode, activation, activation_kw,
            *args, **kwargs
        )
        self._stem = nn.Sequential(
            Conv2dBlock(
                in_channels, 64, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                64, 64, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
        )
