from typing import Dict, Sequence, Tuple, Union

from kaitorch.typing import real
from tab import TAB


def format(
    frames: Sequence[TAB.Frame],
    x: Sequence[Sequence[Tuple[str, Sequence[Tuple[real, real]]]]]
) -> Sequence[
    Sequence[Dict[str, Union[str, Sequence[Tuple[real, real]]]]]
]:
    outs = []
    for f, pred in zip(frames, x):
        bounds = []
        for (cat, points) in pred:
            bounds.append(
                {
                    'semantics': cat,
                    'points': points
                }
            )
        outs.append(
            {
                'sequence': f.sequence,
                'id': f.id,
                'boundaries': bounds
            }
        )

    return outs
