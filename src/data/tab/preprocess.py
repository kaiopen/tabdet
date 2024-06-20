from typing import Union
from pathlib import Path

from tqdm import tqdm
import torch

from kaitorch.typing import TorchTensorLike, Int
from kaitorch.data import Group, ReverseGroup, \
    cell_from_size, squared_euclidean_distance
from tab import Evaluator, Sampler, TAB


def preprocess(
    root: Union[str, Path],
    split: str = 'train',
    size: TorchTensorLike[Int] = (256, 512),
    scale: int = 1,
    resample: bool = False,
    dst: Union[Path, str] = Path.cwd().joinpath('tmp', 'TAB'),
    *args, **kwargs
) -> None:
    r'''Preprocess for training.

    ### Args:
        - root: path to the TAB dataset.
        - split: "train", "val" or "test"
        - size: size of the input BEV images. It should be divided exactly by
            the `scale`.
        - scale
        - resample: resample keypoints of boundaries.

    '''
    tab = TAB(root, split)

    lower_bound = torch.as_tensor((TAB.RANGE_X[0], TAB.RANGE_Y[0]))
    upper_bound = torch.as_tensor((TAB.RANGE_X[1], TAB.RANGE_Y[1]))
    size = torch.as_tensor(size)
    cell = cell_from_size(lower_bound, upper_bound, size)
    group = Group(
        lower_bound, cell, upper_bound=upper_bound, return_offset=True
    )
    reverse = ReverseGroup(lower_bound, cell)

    size //= scale
    sampler = Sampler(size)

    h, w = size.tolist()
    size_hm = (h, w, len(TAB.SEMANTICS))  # the size of the target heatmap
    size_off = (h, w, 2)  # the size of the target offset map.
    # Indices of the anchors in the target heatmap. (HW, 2)
    indices = torch.stack(
        torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'),
        dim=-1
    ).reshape(-1, 2)
    # Anchors in the target heatmap. (HW, 1, 2)
    anchors = reverse((indices + 0.5) * scale).unsqueeze_(1)

    if isinstance(dst, str):
        dst = Path(dst)

    for f in tqdm(tab):
        # Prepare boundaries for generating heatmaps.
        # objects = []

        hm = torch.zeros(size_hm)
        off = torch.zeros(size_off)

        for bound in tab.get_boundaries(f):
            c = TAB.SEMANTICS.index(bound['semantics'])

            if resample or 'keypoints' not in bound:
                bound['keypoints'] = sampler(bound['points'])

            points, radii = Evaluator.decode_keypoints(bound)
            # radii *= 2  # more foreground anchors

            # Distances between anchors and keypoints. (num_anc, num_point)
            ds = squared_euclidean_distance(anchors, points.unsqueeze(0))
            # (num_anchor, num_point)
            mask = ds < torch.pow(radii, 2).unsqueeze_(0)
            ds[torch.logical_not(mask)] = 1e6
            mask = torch.any(mask, dim=1)  # foreground anchors (num_anc,)

            ds, inds = torch.min(ds, dim=1)  # (num_anc,), (num_anc,)
            xy = indices[mask]  # indices for foreground anchors
            x = xy[:, 0]
            y = xy[:, 1]
            hm[x, y, c] = torch.maximum(
                hm[x, y, c],
                torch.exp(
                    -0.5 * ds[mask] / torch.pow(
                        (2 * radii[inds[mask]] + 1) / 6, 2
                    )
                )
            )

            # Do adjustment.
            # It is necessary if the radii is smaller than the size of a grid.
            # Indices of strong foreground anchors (num_key,)
            points_hm = torch.add(*group(points)) / scale
            xy = points_hm.long()
            x = xy[:, 0]
            y = xy[:, 1]
            hm[x, y, c] = 1
            off[x, y] = points_hm - xy

        p = dst.joinpath(f.sequence, f.id + '.pth')
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save((hm.unsqueeze_(0), off.unsqueeze_(0)), p)
