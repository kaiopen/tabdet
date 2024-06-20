from typing import Any, Dict, Sequence, Union

from torch import nn

from kaitorch.nn.conv import Conv2dBlock
from kaitorch.zoo.deeplabv3plus import Block, DeepLabV3Plus as _DeepLabV3Plus


class DeepLabV3Plus(_DeepLabV3Plus):
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
                in_channels, 32, 3, 1, 1,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                32, 64, 3, 1, 1,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._block_1 = Block(
            64, 128,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw,
            num=2,
            activate_first=False,
            grow_first=True
        )
