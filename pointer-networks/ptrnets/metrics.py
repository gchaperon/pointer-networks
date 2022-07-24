# TODO: maybe rename to metrics.py?
import math
import random
import typing as tp
import functools
import torch
from torch.nn.utils.rnn import PackedSequence

import torchmetrics
import shapely.geometry

_Point = tp.Tuple[float, float]


def _multiarange(counts: torch.Tensor) -> torch.Tensor:
    """Returns a sequence of aranges concatenated along the first dimension.

    >>> counts = torch.tensor([1, 3, 2])
    >>> _multiarange(counts)
    torch.tensor([0, 0, 1, 2, 0, 1])

    """
    counts1 = counts[:-1]
    reset_index = counts1.cumsum(0)

    incr = torch.ones(int(counts.sum()), dtype=torch.int64)
    incr[0] = 0
    incr[reset_index] = 1 - counts1
    out: torch.Tensor = incr.cumsum(0)
    return out


class TokenAccuracy(torchmetrics.Metric):
    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: PackedSequence, target: PackedSequence) -> None:
        if prediction.data.ndim == 2:
            prediction._replace(data=prediction.data.argmax(1))
        # prediction and target should be padded to the same length so the shapes match
        pad_length = max(len(prediction.batch_sizes), len(target.batch_sizes))
        prediction_padded, prediction_lens = torch.nn.utils.rnn.pad_packed_sequence(
            prediction, batch_first=True, total_length=pad_length
        )
        target_padded, target_lens = torch.nn.utils.rnn.pad_packed_sequence(
            target, batch_first=True, total_length=pad_length
        )

        # correct only among target tokens, if prediction is longer extra tokens are
        # ignored
        selection = (torch.repeat_interleave(target_lens), _multiarange(target_lens))
        self.correct += torch.sum(
            prediction_padded[selection] == target_padded[selection]
        )
        self.total += torch.sum(target_lens)

    def compute(self) -> torch.Tensor:
        if self.correct == 0:
            return torch.tensor(0.)
        return self.correct / self.total  # type:ignore[operator]


class SequenceAccuracy(torchmetrics.Metric):
    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: PackedSequence, target: PackedSequence) -> None:
        if prediction.data.ndim == 2:
            prediction._replace(data=prediction.data.argmax(1))
        # prediction and target should be padded to the same length so the shapes match
        pad_length = max(len(prediction.batch_sizes), len(target.batch_sizes))
        prediction_padded, prediction_lens = torch.nn.utils.rnn.pad_packed_sequence(
            prediction, batch_first=True, total_length=pad_length
        )
        target_padded, target_lens = torch.nn.utils.rnn.pad_packed_sequence(
            target, batch_first=True, total_length=pad_length
        )

        batch_size = target_padded.shape[0]
        self.correct += torch.sum(torch.all(prediction_padded == target_padded, dim=1))
        self.total += batch_size  # type:ignore[operator]

    def compute(self) -> torch.Tensor:
        if self.correct == 0:
            return torch.tensor(0.)
        return self.correct / self.total  # type:ignore[operator]


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


def token_accuracy(
    prediction: PackedSequence,
    target: PackedSequence,
) -> torch.Tensor:
    # prediction shape: (L, B, C)
    # target shape: (L, B)
    correct: torch.Tensor = prediction.data.argmax(1) == target.data
    return correct.sum() / len(correct)


def sequence_accuracy(
    prediction: PackedSequence, target: PackedSequence
) -> torch.Tensor:
    # prediction shape: (L, B, C)
    # target shape: (L, B)
    pad = torch.nn.utils.rnn.pad_packed_sequence
    prediction_padded, _ = pad(prediction)
    target_padded, _ = pad(target)
    correct: torch.Tensor = (prediction_padded.argmax(2) == target_padded).all(dim=0)
    return correct.sum(), len(correct)


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
            point_set[target[: target_len - 1] - 1].tolist()
        )
        predicted_polygon = shapely.geometry.Polygon(
            point_set[prediction[: prediction_len - 1] - 1].tolist()
        )
        correct.append(target_polygon.equals(predicted_polygon))

    return sum(correct) / len(correct)


class PolygonAccuracy(torchmetrics.Metric):
    def __init__(self):
        pass

    def update(self, prediction, target):
        pass

    def compute(self):
        pass


def area_coverages(
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
            point_set[target[: target_len - 1] - 1].tolist()
        )
        predicted_polygon = shapely.geometry.Polygon(
            point_set[prediction[: prediction_len - 1] - 1].tolist()
        )
        if predicted_polygon.is_simple:
            coverages.append(predicted_polygon.area / target_polygon.area)
        else:
            coverages.append(-1.0)

    return torch.tensor(coverages)


class AreaCoverage(torchmetrics.Metric):
    def __init__(self):
        pass

    def update(self, prediction, target):
        pass

    def compute(self):
        pass
