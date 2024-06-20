import torch

from kaitorch.typing import TorchTensor, TorchBool, TorchInt64, TorchReal, real
from kaitorch.data import squared_euclidean_distance


def deep_first_search_(
    index: int,
    id_cluster: int,
    mask_neighbor: TorchTensor[TorchBool],
    ret: TorchTensor[TorchInt64]
):
    r'''Search neighbors by deep first search.

    ### Args:
        - index: index of a point. The neighbors of it will be searched.
        - id_cluster: ID of a cluster. The neighbors will be assigned the ID.
        - mask_neighbor: mask indicating neighbors. Its shape should be
            `(N, N)`, where `N` is the number of the points.
        - ret: result of clustering.

    '''
    indices = torch.nonzero(mask_neighbor[index], as_tuple=True)[0]
    indices = indices[-1 == ret[indices]]  # unvisited neighbors
    ret[indices] = id_cluster
    for ind in indices.tolist():
        deep_first_search_(ind, id_cluster, mask_neighbor, ret)


class DBSCAN:
    def __init__(
        self,
        radius: real,
        num_sample: int = 4,
        *args, **kwargs
    ) -> None:
        r'''
        The neighbors of a core included the core itself and the neighbors'
        neighbors consist of a cluster.

        ### Args:
            - radius
            - num_sample

        ### Methods:
            - __call__

        __call__
        ### Args:
            - points: n-dim points. Its shape should be `(N, C)`.

        ### Returns:
            - Cluster IDs. Its shape is `(N,)`.

        '''
        self._radius = radius * radius
        self._num_sample = num_sample

    def __call__(
        self, points: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchInt64]:
        mask_neighbor = squared_euclidean_distance(
            points.unsqueeze(1), points.unsqueeze(0)
        ) < self._radius  # (num_point, num_point)

        count = 0
        ret = torch.ones(
            points.shape[0], dtype=torch.int64, device=points.device
        ) * -1
        for ind_core in torch.nonzero(
            torch.sum(mask_neighbor, dim=-1) >= self._num_sample,
            as_tuple=True
        )[0].tolist():
            if -1 == ret[ind_core]:
                ret[ind_core] = count
                deep_first_search_(ind_core, count, mask_neighbor, ret)
                count += 1
        return ret


if '__main__' == __name__:
    a = torch.tensor([[1., 0], [1.5, 0], [1.5, 1], [0, 2]])
    cluster = DBSCAN(radius=1.5, num_sample=1)
    clusters = cluster(a)
    print(clusters)
