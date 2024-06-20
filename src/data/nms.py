from kaitorch.typing import TorchTensor, TorchFloat, TorchInt64


class NMS:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(
        self, scores: TorchTensor[TorchFloat], points: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchInt64]:
        raise NotImplementedError
