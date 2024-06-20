from typing import Any, Dict, Sequence, Tuple, Union

from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.nn.conv import Conv2dBlock


class Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_category: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        r'''A boundary detector with an inner associative embedding.

        ### Args:
            - in_channels
            - num_category
            - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or
                their combination working on dimension `H` and `W` of the
                input. Its length should be less than or equal to 2 if it is a
                sequence.
            - activation: `relu`, `leakyrelu` or other activation.
            - activation_kw: arguments of activation.

        ### Methods:
            - forward

        forward
        ### Args:
            - x: feature map. Its shape should be `(B, in_channels, H, W)`.

        ### Returns:
            - A feature map for categories. Its shape is
                `(B, num_category, H, W)`.
            - A feature map for offsets. Its shape is `(B, 2, H, W)`.

        '''
        super().__init__()
        self._cat = nn.Sequential(
            Conv2dBlock(
                in_channels, in_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                in_channels, in_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            nn.Conv2d(in_channels, num_category, 1)
        )

        self._off = nn.Sequential(
            Conv2dBlock(
                in_channels, in_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                in_channels, in_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            nn.Conv2d(in_channels, 2, 1)
        )

    def forward(
        self, x: TorchTensor[TorchFloat]
    ) -> Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]:
        return self._cat(x), self._off(x)
