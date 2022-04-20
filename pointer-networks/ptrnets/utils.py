import math
import random
import typing as tp
import functools
import torch
from torch.nn.utils.rnn import PackedSequence

import shapely.geometry

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


@tour_distance.register
def _(points: PackedSequence, indices: PackedSequence) -> torch.Tensor:
    points_unpacked, point_sets_lens = torch.nn.utils.rnn.pad_packed_sequence(points)
    # pad with ones so the last point gets repeated when selecting
    indices_unpacked, indices_lens = torch.nn.utils.rnn.pad_packed_sequence(
        indices, padding_value=1
    )
    batch_arange = torch.arange(points_unpacked.shape[1], device=points_unpacked.device)
    curr = points_unpacked[indices_unpacked[:-1] - 1, batch_arange]
    next_ = points_unpacked[indices_unpacked[1:] - 1, batch_arange]
    return torch.sum(torch.sqrt(torch.sum((next_ - curr) ** 2, dim=2)), dim=0)


def random_solve(n_points: int) -> tp.List[int]:
    return [1, *random.sample(range(2, n_points + 1), n_points - 1), 1]


def token_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    # prediction shape: (L, B, C)
    # target shape: (L, B)
    if isinstance(prediction, PackedSequence) and isinstance(target, PackedSequence):
        correct = prediction.data.argmax(1) == target.data
    else:
        correct = prediction.flatten(0, 1).argmax(1) == target.flatten()
    return correct.sum() / len(correct)


def sequence_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    # prediction shape: (L, B, C)
    # target shape: (L, B)
    if isinstance(prediction, PackedSequence) and isinstance(target, PackedSequence):
        unpack = torch.nn.utils.rnn.pad_packed_sequence
        prediction, _ = unpack(prediction)
        target, _ = unpack(target)
    prediction = prediction.argmax(2)
    correct = (prediction == target).all(dim=0)
    return correct.sum() / len(correct)


def polygon_accuracy(
    point_sets: PackedSequence, predictions: PackedSequence, targets: PackedSequence
) -> float:
    pad = functools.partial(torch.nn.utils.rnn.pad_packed_sequence, batch_first=True)

    correct = []
    for (
        point_set,
        point_set_len,
        prediction,
        prediction_len,
        target,
        target_len,
    ) in zip(*pad(point_sets), *pad(predictions), *pad(targets)):
        target_polygon = shapely.geometry.Polygon(
            point_set[target[: target_len - 1] - 1]
        )
        predicted_polygon = shapely.geometry.Polygon(
            point_set[prediction[: prediction_len - 1] - 1]
        )
        correct.append(target_polygon.equals(predicted_polygon))

    return sum(correct) / len(correct)


def area_coverage(
    point_sets: PackedSequence, predictions: PackedSequence, targets: PackedSequence
) -> torch.Tensor:
    pad = functools.partial(torch.nn.utils.rnn.pad_packed_sequence, batch_first=True)
    coverages = []
    for (
        point_set,
        point_set_len,
        prediction,
        prediction_len,
        target,
        target_len,
    ) in zip(*pad(point_sets), *pad(predictions), *pad(targets)):
        target_polygon = shapely.geometry.Polygon(
            point_set[target[: target_len - 1] - 1]
        )
        predicted_polygon = shapely.geometry.Polygon(
            point_set[prediction[: prediction_len - 1] - 1]
        )
        if predicted_polygon.is_simple:
            coverages.append(predicted_polygon.area / target_polygon.area)
        else:
            coverages.append(-1)

    return torch.tensor(coverages)
