from torch.nn import Identity

from kaitorch.zoo.deeplabv3plus import DeepLabV3Plus
from kaitorch.zoo.hrnet import HRNet
from kaitorch.zoo.unet import UNet

# from .deeplabv3plus import DeepLabV3Plus
# from .hrnet import HRNet


BACKBONE = {
    'Identity': Identity,
    'UNet': UNet,
    'HRNet': HRNet,
    'DeepLabV3Plus': DeepLabV3Plus
}
