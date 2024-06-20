from torch.nn import Identity

from kaitorch.zoo.hrnet import Neck

from .deeplabv3plus import DeepLabV3Plus


NECK = {
    'Identity': Identity,
    'HRNet': Neck,
    'DeepLabV3Plus': DeepLabV3Plus
}
