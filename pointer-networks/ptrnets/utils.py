import math
import random
import typing as tp
import functools
import torch

_Point = tp.Tuple[float, float]


def distance(pt1: _Point, pt2: _Point) -> float:
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


@functools.singledispatch
def tour_distance(points, indices):
    raise NotImplementedError(
        f"Not implemented for arg `points` of type {type(points)}"
    )


@tour_distance.register(list)
def _(points: tp.List[_Point], indices: tp.List[int]) -> float:
    dist = 0.0
    # NOTE: indices are 1-indexed
    indices = [i - 1 for i in indices]
    for current, next in zip(indices[:-1], indices[1:]):
        dist += distance(points[current], points[next])
    return dist


@tour_distance.register
def _(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Computes batched tour distance.
    points should have dims (tour_len, batch, 2)
    indices should have dims (tour_len + 1, batch)
    """
    batch_arange = torch.arange(points.shape[1], device=points.device)
    # NOTE: indices are 1-indexed
    curr = points[indices[:-1] - 1, batch_arange]
    next_ = points[indices[1:] - 1, batch_arange]
    return torch.sum(torch.sqrt(torch.sum((next_ - curr) ** 2, dim=2)), dim=0)


def random_solve(n_points: int) -> tp.List[int]:
    return [1, *random.sample(range(2, n_points + 1), n_points - 1), 1]


def token_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    # prediction shape: (L, B, C)
    # target shape: (L, B)
    correct = prediction.flatten(0, 1).argmax(1) == target.flatten()
    return correct.sum() / len(correct)


def sequence_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    # prediction shape: (L, B, C)
    # target shape: (L, B)
    prediction = prediction.argmax(2)
    correct = (prediction == target).all(dim=0)
    return correct.sum() / len(correct)
