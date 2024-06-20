import torch

from kaitorch.utils import Configer
from tab import TAB


def set_default_config_(cfg: Configer) -> Configer:
    range_xy = cfg.data.range_xy
    if 'default' == range_xy:
        cfg.data.range_xy = torch.as_tensor(
            list(TAB.RANGE_X) + list(TAB.RANGE_Y)
        )
    else:
        cfg.data.range_xy = torch.as_tensor(range_xy)

    if 'default' == cfg.data.num_category:
        cfg.data.num_category = len(TAB.SEMANTICS)

    if 'default' == cfg.data.categories:
        cfg.data.categories = TAB.SEMANTICS

    return cfg
