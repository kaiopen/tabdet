from typing import Any, Dict
from torch import nn

from .bev import BEV
from .backbone import BACKBONE
from .neck import NECK
from .head import HEAD


class Model(nn.Module):
    def __init__(
        self,
        bev: Dict[str, Any],
        backbone: Dict[str, Any],
        neck: Dict[str, Any],
        head: Dict[str, Any],
        *args, **kwargs
    ) -> None:
        r'''
        Combination of pillar map network, backbone, neck and head.

        ### Args:
            - bev: parameters for BEV generator.
            - backbone: parameters for backbone.
            - neck: parameters for neck.
            - head: parameters for head.

        ### Methods;
            - forward

        '''
        super().__init__()
        self.bev = BEV[bev.pop('type')](**bev)
        self.backbone = BACKBONE[backbone.pop('type')](**backbone)
        self.neck = NECK[neck.pop('type')](**neck)
        self.head = HEAD[head.pop('type')](**head)

    def forward(self, x):
        return self.head(self.neck(self.backbone(self.bev(x))))
