import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import pytorch_lightning as pl
import ptrnets
import functools
import operator
import typing as tp
import torchmetrics
import ptrnets.metrics as metrics
import dataclasses


# Some utilities
# =====================================================
def _prepend(sequences: PackedSequence, tensor: torch.Tensor) -> PackedSequence:
    """Prepends a tensor to each sequence"""
    padded, lens = nn.utils.rnn.pad_packed_sequence(sequences)
    # repeat tensor batch_size times
    # tensor shape should be the same shape as each token in a sequence
    batch_size = padded.shape[1]
    padded = torch.cat(
        [tensor.repeat(1, batch_size, *[1] * (padded.ndim - 2)), padded], dim=0
    )
    return nn.utils.rnn.pack_padded_sequence(
        padded, lengths=lens + 1, enforce_sorted=False
    )


def _cat_packed_sequences(packed_sequences: tp.List[PackedSequence]) -> PackedSequence:
    """Concatenate packed sequences along batch dimention"""
    max_sequence_len = max(len(packed.batch_sizes) for packed in packed_sequences)
    padded, lens = zip(
        *(
            nn.utils.rnn.pad_packed_sequence(packed, total_length=max_sequence_len)
            for packed in packed_sequences
        )
    )
    concatenated = nn.utils.rnn.pack_padded_sequence(
        torch.cat(padded, dim=1), torch.cat(lens), enforce_sorted=False
    )
    return concatenated


# Model definition
# =====================================================
class Attention(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        # NOTE: Naming convention follows the paper
        self.activation = nn.Tanh()
        self.W1 = nn.Parameter(torch.empty(input_size, input_size))
        self.W2 = nn.Parameter(torch.empty(input_size, input_size))
        self.v = nn.Parameter(torch.empty(input_size))

    def forward(
        self,
        encoder_output: tp.Union[torch.Tensor, PackedSequence],
        decoder_output: tp.Union[torch.Tensor, PackedSequence],
    ) -> PackedSequence:
        # treat everything as PackedSequence
        if isinstance(encoder_output, torch.Tensor):
            encoder_output = nn.utils.rnn.pack_sequence(encoder_output.unbind(1))
        if isinstance(decoder_output, torch.Tensor):
            decoder_output = nn.utils.rnn.pack_sequence(decoder_output.unbind(1))

        assert (
            encoder_output.batch_sizes[0] == decoder_output.batch_sizes[0]
        ), "batch_size missmatch"

        encoder_output = encoder_output._replace(data=encoder_output.data @ self.W1)
        decoder_output = decoder_output._replace(data=decoder_output.data @ self.W2)
        # shape: (max_enc_seq_len, batch, hidden)
        encoder_unpacked, encoder_lens = nn.utils.rnn.pad_packed_sequence(
            encoder_output
        )
        # shape: (max_dec_seq_len, batch, hidden)
        decoder_unpacked, decoder_lens = nn.utils.rnn.pad_packed_sequence(
            decoder_output
        )
        # shape: (max_dec_seq_len, max_enc_sec_len, batch)
        scores = (
            self.activation(decoder_unpacked.unsqueeze(1) + encoder_unpacked) @ self.v
        )
        # mask padded positions in dim max_enc_sec_len
        max_enc_len = len(encoder_output.batch_sizes)
        batch_size = encoder_output.batch_sizes[0]
        scores[
            :,
            torch.arange(max_enc_len)[:, None].expand(max_enc_len, batch_size)
            >= encoder_lens,
        ] = -torch.inf
        return nn.utils.rnn.pack_padded_sequence(
            scores.transpose(1, 2), lengths=decoder_lens, enforce_sorted=False
        )


class PointerNetwork(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        learn_rate: float,
        init_range: tp.Tuple[float, float],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learn_rate = learn_rate
        self.init_range = init_range

        # => in the paper
        self.start_symbol = nn.Parameter(torch.empty(input_size))
        # <= in the paper
        self.end_symbol = nn.Parameter(torch.empty(hidden_size))
        # learn initial cell state
        self.encoder_c_0 = nn.Parameter(torch.empty(hidden_size))
        # Modules
        # ======================================================
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=False,
        )
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=False,
        )
        self.attention = Attention(input_size=hidden_size)
        self.reset_parameters()

        # Metrics
        # ======================================================
        metric_collection = torchmetrics.MetricCollection(
            {
                "token_accuracy": metrics.TokenAccuracy(),
                "sequence_accuracy": metrics.SequenceAccuracy(),
            }
        )
        # train
        self.train_metrics = metric_collection.clone(prefix="train/")
        self.val_metrics = metric_collection.clone(prefix="val/")

        self.test_tf_metrics = metric_collection.clone(prefix="test/tf_")
        self.test_greedy_metrics = metric_collection.clone(prefix="test/greedy_")
        self.test_beam_metrics = metric_collection.clone(prefix="test/beam_")

    def reset_parameters(self) -> None:
        for param in self.parameters():
            nn.init.uniform_(param, *self.init_range)

    def forward(
        self, encoder_input: PackedSequence, decoder_input: PackedSequence
    ) -> PackedSequence:
        encoder_output, encoder_last_state = self._encoder_forward(encoder_input)
        decoder_output, _ = self.decoder(
            _prepend(decoder_input, self.start_symbol), encoder_last_state
        )
        scores: PackedSequence = self.attention(encoder_output, decoder_output)
        return scores

    def _encoder_forward(
        self, encoder_input: PackedSequence
    ) -> tp.Tuple[PackedSequence, tp.Tuple[torch.Tensor, torch.Tensor]]:
        # expand encoder init state to match LSTM signature
        batch_size = encoder_input.batch_sizes[0]
        encoder_init_state = (
            self.end_symbol.repeat(1, batch_size, 1),
            self.encoder_c_0.repeat(1, batch_size, 1),
        )
        encoder_output, encoder_last_state = self.encoder(
            encoder_input, encoder_init_state
        )
        return _prepend(encoder_output, self.end_symbol), encoder_last_state

    def training_step(self, batch: ptrnets.data._Batch, batch_idx: int) -> torch.Tensor:
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)
        loss = self._get_loss(prediction, target)

        self.log_dict(
            {
                "train/loss": loss,
                **self.train_metrics(prediction, target),
            },
            batch_size=target.batch_sizes[0],
        )
        return loss

    def _get_loss(
        self, prediction: PackedSequence, target: PackedSequence
    ) -> torch.Tensor:
        return F.cross_entropy(prediction.data, target.data)

    def validation_step(
        self, batch: ptrnets.data._Batch, batch_idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)

        self.log_dict(
            {
                "val/loss": self._get_loss(prediction, target),
                **self.val_metrics(prediction, target),
            },
            batch_size=target.batch_sizes[0],
        )

    def test_step(
        self, batch: ptrnets.data._Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """test_step logs the same as val_step, and returns the necesary data
        for test_epoch_end to do the decoding. Test epoch end will take quite
        some time."""
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)

        greedy_decoded = self.batch_greedy_decode(encoder_input)
        beam_decoded = self.single_beam_search(encoder_input)
        self.log_dict(
            {
                **self.test_tf_metrics(prediction, target),
                **self.test_greedy_metrics(greedy_decoded, target),
                **self.test_beam_metrics(beam_decoded, target),
            },
            batch_size=target.batch_sizes[0],
        )


    def _test_epoch_end(self, test_step_outputs):
        all_greedy_decoded, all_prediction, all_targets = (
            _cat_packed_sequences(items) for items in zip(*test_step_outputs)
        )
        pad = functools.partial(
            nn.utils.rnn.pad_packed_sequence,
            batch_first=True,
            total_length=all_greedy_decoded.batch_sizes.shape[0],
            # total_length=encoder_input.batch_sizes.shape[0] + 2,
        )
        all_greedy_decoded = pad(all_greedy_decoded)[0]
        all_prediction = pad(all_prediction)[0]
        all_targets = pad(all_targets)[0]

        pass

    def _test_step_old(
        self, batch: ptrnets.data._Batch, batch_idx: int
    ) -> tp.Tuple[PackedSequence, PackedSequence, PackedSequence]:
        encoder_input, _, target = batch
        encoder_input_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(
            encoder_input
        )
        decoded = []
        for points_padded, len_ in zip(encoder_input_unpacked.unbind(1), lens_unpacked):
            decoded.append(
                torch.tensor(
                    self.decode(points_padded[:len_]), device=points_padded.device
                )
            )

        return (
            encoder_input,
            nn.utils.rnn.pack_sequence(decoded, enforce_sorted=False),
            target,
        )

    def configure_optimizers(self) -> tp.Dict["str", tp.Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learn_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.96
            ),
        }


@dataclasses.dataclass
class _Beam:
    indices: tp.List[int]
    score: float
    decoder_input: PackedSequence = dataclasses.field(repr=False)
    # see nn.LSTM docs for `last_hidden` type
    last_hidden: tp.Tuple[torch.Tensor, torch.Tensor] = dataclasses.field(repr=False)
    maxlen: tp.Optional[int] = None

    def __post_init__(self) -> None:
        """There are some a cases where the beam is finished but is in a "inconsistent"
        state, like being to long but not ending in 0 or ending in 0 but without having
        returned to the first decoded index. To keep consistency of the format i'm
        taking some measures here."""
        if self.is_done() and self.indices[-1] != 0:
            self.indices.append(0)
        if self.is_done() and self.indices[-2] != self.indices[0]:
            self.indices.insert(-1, self.indices[0])

    def is_done(self) -> bool:
        # NOTE: this is_done logic only applies to convex hull
        ends_with_zero = len(self.indices) > 0 and self.indices[-1] == 0
        has_looped = len(self.indices) > 1 and self.indices[0] == self.indices[-1]
        is_too_long = len(self.indices) > self.maxlen if self.maxlen else False
        return ends_with_zero or has_looped or is_too_long


def _unravel_index(
    indices: torch.Tensor, shape: tp.Tuple[int, ...]
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    # Supper innefficient to copy to cpu and then back to cuda if indices
    # is a cuda tensor, but for now it suffices.
    device = indices.device
    unraveled_coords = np.unravel_index(indices.cpu().numpy(), shape)
    return tuple(torch.tensor(arr, device=device) for arr in unraveled_coords)


class PointerNetworkForConvexHull(PointerNetwork):
    @torch.no_grad()
    def _old_decode(
        self,
        input: torch.Tensor,
        nbeams: int = 3,
        maxlen: tp.Optional[int] = None,
    ) -> tp.List[int]:
        assert input.ndim == 2, "input should be a 2 dim tensor, a sequence of points"
        maxlen = maxlen or input.shape[0] + 2

        encoder_init_state = (
            self.end_symbol.view(1, 1, -1),
            self.encoder_c_0.view(1, 1, -1),
        )
        encoder_output, encoder_last_hidden = self.encoder(
            nn.utils.rnn.pack_sequence([input]), encoder_init_state
        )
        encoder_output = _prepend(encoder_output, self.end_symbol)

        beams: tp.List[_Beam] = [
            _Beam(
                indices=[],
                score=0.0,
                decoder_input=nn.utils.rnn.pack_sequence(
                    [self.start_symbol.unsqueeze(0)]
                ),
                last_hidden=encoder_last_hidden,
                maxlen=maxlen,
            )
        ]

        while not all(beam.is_done() for beam in beams):
            candidates: tp.List[_Beam] = []
            for beam in beams:
                if beam.is_done():
                    candidates.append(beam)
                    continue

                decoder_output, decoder_last_hidden = self.decoder(
                    beam.decoder_input, beam.last_hidden
                )
                # pad_packed_sequence returns a tuple: the padded tensor and the
                # original lengths. The padded tensor has shape [1, 1, L], because it
                # has a batch size of 1 and is a single token for which to compute the
                # attention scores, so I select using [0][0, 0]
                attention_scores = nn.utils.rnn.pad_packed_sequence(
                    self.attention(encoder_output, decoder_output)
                )[0][0, 0]
                # mask invalid values
                # if indices is not a valid polygon (e.g. less than 3 points), mask
                # all previous values so a new one must be produced.
                # when indices is a valid polygon, mask everything except first value so
                # that no previous point can be decoded, except for the first one which
                # means closing the polygon.
                mask = [0, *beam.indices] if len(beam.indices) < 3 else beam.indices[1:]
                attention_scores[
                    torch.tensor(mask, device=attention_scores.device)
                ] = float("-inf")
                probs, indices = attention_scores.softmax(dim=0).sort(descending=True)
                for prob, index in zip(probs[:nbeams], indices[:nbeams]):
                    candidates.append(
                        _Beam(
                            indices=[*beam.indices, index.item()],
                            score=beam.score - torch.log(prob).item(),
                            decoder_input=nn.utils.rnn.pack_sequence(
                                [input[None, index - 1]]
                            ),
                            last_hidden=decoder_last_hidden,
                            maxlen=maxlen,
                        )
                    )
                beams = sorted(candidates, key=operator.attrgetter("score"))[:nbeams]

        # NOTE: remove 0 index at the end to match :class:`ptrnsets.data.ConvexHull`
        # format
        return beams[0].indices[:-1]

    @torch.no_grad()
    def batch_greedy_decode(self, inputs: PackedSequence) -> PackedSequence:
        """inputs of shape (B, L_in*, 2), outuput of shape (B, L_out*), both as
        packed sequence

        in cpu: This works exactly the same as teacher forcing, which is
        expected.  in cuda: for some reason there are differences between the
        results from this fun and teacher forcing. They don't occur in cpu, so
        I asume the cuda -> python communication at each steps messes things
        up, whereas when using teacher forcing, all steps occur in a highly
        optimized cudnn kernel. But who knows really...
        """
        # +2 because solution has to loop and then add 0
        max_len = len(inputs.batch_sizes) + 2
        batch_size = inputs.batch_sizes[0]

        encoder_output, last_hidden = self._encoder_forward(inputs)

        decoder_input = self.start_symbol.repeat(1, batch_size, 1)

        predictions = []

        for _ in range(max_len):
            _, last_hidden = self.decoder(decoder_input, last_hidden)

            scores = self.attention(
                encoder_output, nn.utils.rnn.pack_sequence(last_hidden[0].unbind(1))
            )
            scores_padded, _ = nn.utils.rnn.pad_packed_sequence(scores)
            indices = scores_padded.argmax(2)
            decoder_input = nn.utils.rnn.pad_packed_sequence(inputs)[0][
                indices - 1, torch.arange(batch_size)
            ]
            predictions.append(indices)
            if (torch.cat(predictions) == 0).any(0).all():
                break
        else:
            # if decoding finished without breaking, add prediction of only
            # zeroes to keep format consistent
            predictions.append(torch.zeros_like(predictions[0]))

        predictions = torch.cat(predictions)
        # find indices of
        _, lens = torch.min(predictions, dim=0)
        # the docs say that lens should be in cpu, add 1 because we want to keep
        # the last 0 to have the same format as what comes out of the forward pass
        return nn.utils.rnn.pack_padded_sequence(
            predictions, lens.cpu() + 1, enforce_sorted=False
        )

    @torch.no_grad()
    def single_beam_search(
        self, inputs: PackedSequence, nbeams: int = 5
    ) -> PackedSequence:
        """Performs beam search with a single sequence. Beams are processed in
        parallel."""

        # +2 because solution has to loop and then add 0
        device = inputs.data.device
        max_len = len(inputs.batch_sizes) + 2
        batch_size = inputs.batch_sizes[0]

        assert batch_size == 1
        # repeat inputs for each beam
        input_padded, input_lens = nn.utils.rnn.pad_packed_sequence(inputs)
        inputs = nn.utils.rnn.pack_padded_sequence(
            input_padded.repeat(1, nbeams, 1),
            input_lens.repeat(nbeams),
            enforce_sorted=False,
        )

        # state that should be update after each recurrent step
        # last_hidden
        # decoder_input
        encoder_output, last_hidden = self._encoder_forward(inputs)

        decoder_input = self.start_symbol.repeat(1, nbeams, 1)

        beams = torch.empty(0, nbeams, device=device)
        beam_scores = torch.tensor([0.0, *[torch.inf] * (nbeams - 1)], device=device)

        for _ in range(max_len):
            _, last_hidden = self.decoder(decoder_input, last_hidden)
            # for each beam, next token scores
            # (nbeams, max_input_len + 1)
            logits = nn.utils.rnn.pad_packed_sequence(
                self.attention(encoder_output, last_hidden[0])
            )[0].squeeze(0)
            finished_mask = (beams == 0).any(0)
            logits[finished_mask, 1:] = -torch.inf
            probs = logits.softmax(1)
            # replace -inf supe negative value, but not -inf
            temp_scores = torch.log(probs)
            temp_scores[temp_scores == -torch.inf] = torch.finfo().min
            new_beam_scores = beam_scores[:, None] - temp_scores
            topk_scores, indices = torch.topk(
                new_beam_scores.view(-1), k=nbeams, largest=False
            )
            beams_index, index_prediction = _unravel_index(
                indices, shape=new_beam_scores.shape
            )

            beams = torch.vstack(
                [
                    beams[:, beams_index],
                    index_prediction,
                ]
            )
            beam_scores = topk_scores

            # update recurrent state
            # choose the last hidden from the correct beams
            last_hidden = tuple(t[:, beams_index] for t in last_hidden)
            assert torch.all(
                index_prediction.cpu() < input_lens + 1
            )  # input_lens is always on cpu
            decoder_input = input_padded[index_prediction - 1].transpose(0, 1)
            if torch.all(torch.any(beams == 0, dim=0)):
                break
        else:
            # if no break occurs, add last 0 to al predictions
            beams = torch.vstack([beams, torch.zeros(nbeams, dtype=int, device=device)])

        assert beam_scores.argmin() == 0
        winner = beams[:, 0:1]  # since topk is sorted winner is always first beam
        return nn.utils.rnn.pack_sequence(winner[: winner.argmin() + 1].unbind(1))


class PointerNetworkForTSP(PointerNetwork):
    """Implements a single method, `decode` where logic for TSP is applied.
    This means, all points must be included in the results, and no point can appear
    twice in the response. Uses beam search plus these contraints"""

    @torch.no_grad()
    def _beam_search(self, input: torch.Tensor, nbeams: int) -> tp.List[_Beam]:
        encoder_output, encoder_last_hidden = self.encoder(input.unsqueeze(1))
        encoder_output = _prepend_eos_token(encoder_output)

        beams: tp.List[_Beam] = [
            _Beam(
                indices=[],
                score=0.0,
                decoder_input=torch.ones(2, device=encoder_output.device) * -1,
                last_hidden=encoder_last_hidden,
            )
        ]
        while not all(beam.is_done() for beam in beams):
            candidates: tp.List[_Beam] = []
            for beam in beams:
                # always add finished beams as candidates
                # this simplifies the code (I think), but should be irrelevant
                # for the end result
                if beam.is_done():
                    candidates.append(beam)
                    continue

                # finish decoding when all nodes have been visited
                # go back to first decoded node and append 0 to indicate its done
                if set(beam.indices) == set(range(1, len(input) + 1)):
                    candidates.append(
                        dataclasses.replace(
                            beam, indices=beam.indices + [beam.indices[0], 0]
                        )
                    )
                    continue

                _, (h_n, c_n) = self.decoder(
                    beam.decoder_input.view(1, 1, -1), beam.last_hidden
                )
                # select [0, 0] to undo the .view(1, 1, -1) op and get a vector
                attention_scores = self.attention(encoder_output, h_n)[0, 0]
                # mask nodes visited already, plus node "0" which is invalid
                attention_scores[beam.indices + [0]] = float("-inf")
                probs, indices = attention_scores.softmax(dim=0).sort(descending=True)

                for prob, index in zip(probs[:nbeams], indices[:nbeams]):
                    candidates.append(
                        _Beam(
                            indices=[*beam.indices, index.item()],
                            score=beam.score - torch.log(prob).item(),
                            decoder_input=input[index - 1],
                            last_hidden=(h_n, c_n),
                        )
                    )
            beams = sorted(candidates, key=operator.attrgetter("score"))[:nbeams]
        return beams

    @torch.no_grad()
    def decode(
        self, input: torch.Tensor, k: int = 3, nreturn: int = 1, wscores: bool = False
    ) -> tp.Union[
        tp.Union[tp.List[int], tp.Tuple[tp.List[int], float]],
        tp.Union[tp.List[tp.List[int]], tp.List[tp.Tuple[tp.List[int], float]]],
    ]:
        """This is super slow, maybe in the future (maybe) i will try to make
        it faster, possibly parallel or who knows"""
        assert input.ndim == 2, "input should be a 2 dim tensor, a sequence of points"
        nreturn = nreturn or k
        assert nreturn <= k, (
            f"how can i return {nreturn} beams"
            f" when i'm supposed to use only {k} beams!"
            "\nTHINK MARK, THINK!"
        )

        beams: tp.List[_Beam] = self._beam_search(input, nbeams=k)

        # format output depending on options
        if nreturn == 1:
            if wscores:
                return beams[0].indices[:-1], beams[0].score
            else:
                return beams[0].indices[:-1]
        else:
            if wscores:
                return [(beam.indices[:-1], beam.score) for beam in beams[:nreturn]]
            else:
                return [beam.indices[:-1] for beam in beams[:nreturn]]

    def test_epoch_end(
        self,
        test_step_outputs: tp.List[
            tp.Tuple[PackedSequence, PackedSequence, PackedSequence]
        ],
    ) -> None:
        all_point_sets, all_decoded, all_targets = (
            _cat_packed_sequences(items) for items in zip(*test_step_outputs)
        )

        decoded_tour_distance = torch.mean(
            metrics.tour_distance(all_point_sets, all_decoded)
        )
        # remove all trailing 0s in target
        padded, lens = nn.utils.rnn.pad_packed_sequence(all_targets)
        all_targets = nn.utils.rnn.pack_padded_sequence(padded, lens - 1)
        target_tour_distance = torch.mean(
            metrics.tour_distance(all_point_sets, all_targets)
        )
        self.log("test_decoded_tour_distance", decoded_tour_distance)
        self.log("test_target_tour_distance", target_tour_distance)
        self.log(
            "test_tour_distance_diff",
            decoded_tour_distance - target_tour_distance,
        )
