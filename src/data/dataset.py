from typing import Any, Tuple

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        raise NotImplementedError

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    @staticmethod
    def collate(x):
        raise NotImplementedError

    @staticmethod
    def cuda(x):
        raise NotImplementedError
