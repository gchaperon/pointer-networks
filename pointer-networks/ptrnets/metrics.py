import typing as tp
import functools
import torch
from torch.nn.utils.rnn import PackedSequence

import torchmetrics
import shapely.geometry


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
            prediction = prediction._replace(data=prediction.data.argmax(1))
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
            return torch.tensor(0.0)
        return self.correct / self.total  # type:ignore[operator]


class SequenceAccuracy(torchmetrics.Metric):
    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: PackedSequence, target: PackedSequence) -> None:
        if prediction.data.ndim == 2:
            prediction = prediction._replace(data=prediction.data.argmax(1))
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
            return torch.tensor(0.0)
        return self.correct / self.total  # type:ignore[operator]


class PolygonAccuracy(torchmetrics.Metric):
    correct: torch.Tensor
    total: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        pass

    def update(
        self,
        point_sets: PackedSequence,
        predictions: PackedSequence,
        targets: PackedSequence,
    ) -> None:
        pad = functools.partial(
            torch.nn.utils.rnn.pad_packed_sequence, batch_first=True
        )

        if predictions.data.ndim == 2:
            predictions = predictions._replace(data=predictions.data.argmax(1))

        correct, total = 0, 0
        for i, (
            point_set,
            point_set_len,
            prediction,
            prediction_len,
            target,
            target_len,
        ) in enumerate(zip(*pad(point_sets), *pad(predictions), *pad(targets))):
            target_polygon = shapely.geometry.Polygon(
                point_set[target[: target_len - 1] - 1].tolist()
            )
            predicted_polygon = shapely.geometry.Polygon(
                point_set[prediction[: prediction_len - 1] - 1].tolist()
            )
            correct += target_polygon.equals(predicted_polygon)
            total += 1

        self.correct += correct
        self.total += total

    def compute(self) -> torch.Tensor:
        if self.correct == 0:
            return torch.tensor(0.0)
        return self.correct / self.total


class AverageAreaCoverage(torchmetrics.Metric):
    coverages: tp.List[torch.Tensor]
    is_valid: tp.List[torch.Tensor]

    def __init__(self, is_valid_threshold: float = 0.1) -> None:
        super().__init__()

        self.is_valid_threshold = is_valid_threshold
        self.add_state("coverages", default=[], dist_reduce_fx="cat")
        self.add_state("is_valid", default=[], dist_reduce_fx="cat")

    def update(
        self,
        point_sets: PackedSequence,
        predictions: PackedSequence,
        targets: PackedSequence,
    ) -> None:
        pad = functools.partial(
            torch.nn.utils.rnn.pad_packed_sequence, batch_first=True
        )

        coverages: tp.List[float] = []
        is_valid: tp.List[bool] = []
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
            coverages.append(predicted_polygon.area / target_polygon.area)
            is_valid.append(predicted_polygon.is_simple)

        self.coverages.append(torch.tensor(coverages))
        self.is_valid.append(torch.tensor(is_valid))

    def compute(self) -> torch.Tensor:
        is_valid = torch.cat(self.is_valid, dim=0)
        coverages = torch.cat(self.coverages, dim=0)
        if torch.sum(~is_valid) > self.is_valid_threshold * len(is_valid):
            return torch.tensor(-1.0)

        return torch.mean(coverages[is_valid])


class TourDistance(torchmetrics.Metric):
    tour_distances: tp.List[torch.Tensor]

    def __init__(self) -> None:
        super().__init__()

        self.add_state("tour_distances", default=[], dist_reduce_fx="cat")

    def update(self, point_sets: PackedSequence, prediction: PackedSequence) -> None:
        batch_size = point_sets.batch_sizes[0]
        device = point_sets.data.device

        point_sets_padded, npoints = torch.nn.utils.rnn.pad_packed_sequence(
            point_sets, batch_first=True
        )

        prediction_padded, prediction_lens = torch.nn.utils.rnn.pad_packed_sequence(
            prediction, batch_first=True
        )
        max_pred_len = prediction_padded.shape[1]

        batch_arange = torch.arange(batch_size, device=device)
        assert torch.all(
            prediction_padded[batch_arange, prediction_lens - 1] == 0
        ), "all prediction should finish with a 0"
        assert torch.all(
            prediction_padded[batch_arange, prediction_lens - 2]
            == prediction_padded[:, 0]
        ), "all tours should end where they start"
        # pad with the first value, so that summing distances after closing
        # tour doesn't increase the tour distance
        prediction_padded += (
            torch.arange(max_pred_len, device=device).expand_as(prediction_padded)
            >= (prediction_lens.to(device) - 1)[:, None]
        ) * prediction_padded[:, 0:1]
        # NOTE: i just trust from decoding that there are no repeated points
        # and all points are visited
        curr = point_sets_padded[batch_arange[:, None], prediction_padded[:, :-1] - 1]
        next_ = point_sets_padded[batch_arange[:, None], prediction_padded[:, 1:] - 1]
        tour_distances = torch.sum(
            torch.sqrt(torch.sum((next_ - curr) ** 2, dim=2)), dim=1
        )
        self.tour_distances.append(tour_distances)

    def compute(self) -> torch.Tensor:
        all_tour_distances = torch.cat(self.tour_distances)
        return all_tour_distances.mean()
