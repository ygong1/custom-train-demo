from streaming import StreamingDataset
from typing import Callable, Any

class CIFAR10Dataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable
                ) -> None:
        super().__init__(local=None, remote=remote, shuffle=shuffle, batch_size=batch_size)
        self.transforms = transforms

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        x = obj['x']
        y = obj['y']
        return self.transforms(x), y