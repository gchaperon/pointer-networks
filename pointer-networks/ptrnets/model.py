import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import pytorch_lightning as pl
import ptrnets
import functools
import operator
import functools
import typing as tp
import ptrnets.metrics as metrics
import dataclasses


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
        encoder_output: PackedSequence,
        decoder_output: PackedSequence,
    ) -> PackedSequence:
        if isinstance(encoder_output, PackedSequence) and isinstance(
            decoder_output, PackedSequence
        ):
            encoder_output = encoder_output._replace(data=encoder_output.data @ self.W1)
            decoder_output = decoder_output._replace(data=decoder_output.data @ self.W2)
            # shape: (max_dec_seq_len, batch, hidden)
            encoder_unpacked, encoder_lens = nn.utils.rnn.pad_packed_sequence(
                encoder_output
            )
            # shape: (max_enc_seq_len, batch, hidden)
            decoder_unpacked, decoder_lens = nn.utils.rnn.pad_packed_sequence(
                decoder_output
            )
            # TODO: maybe mask padded positions?
            # shape: (max_dec_seq_len, max_enc_sec_len, batch)
            scores = (
                self.activation(decoder_unpacked.unsqueeze(1) + encoder_unpacked)
                @ self.v
            )
            return nn.utils.rnn.pack_padded_sequence(
                scores.transpose(1, 2), lengths=decoder_lens, enforce_sorted=False
            )
        else:
            # encoder_output: (enc_seq_len, batch, hidden)
            # decoder_output: (dec_seq_len, batch, hidden)
            # scores: (dec_seq_len, enc_sec_len, batch)
            scores = (
                self.activation(
                    encoder_output @ self.W1 + (decoder_output @ self.W2).unsqueeze(1)
                )
                @ self.v
            )
            return scores.transpose(1, 2)


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


def _append(sequences: PackedSequence, tensor: torch.Tensor) -> PackedSequence:
    """Appends a tensor to each sequence

    This is trickier because elements should be inserted between the last token and the
    padding section"""
    padded, lens = nn.utils.rnn.pad_packed_sequence(
        sequences, total_length=len(sequences.batch_sizes) + 1
    )
    batch_size = padded.shape[1]
    padded[lens, torch.arange(batch_size)] = tensor.repeat(
        batch_size, *[1] * (padded.ndim - 2)
    )

    return nn.utils.rnn.pack_padded_sequence(
        padded, lengths=lens + 1, enforce_sorted=False
    )


def _prepend_eos_token(encoder_output):
    if isinstance(encoder_output, PackedSequence):
        # shape: (max_enc_seq_len, batch, hidden_size)
        encoder_unpacked, encoder_lens = nn.utils.rnn.pad_packed_sequence(
            encoder_output
        )
        encoder_unpacked = torch.cat(
            [
                torch.zeros(
                    1, *encoder_unpacked.shape[1:], device=encoder_unpacked.device
                ),
                encoder_unpacked,
            ]
        )
        return nn.utils.rnn.pack_padded_sequence(
            encoder_unpacked, lengths=encoder_lens + 1, enforce_sorted=False
        )
    else:
        return torch.cat(
            [
                torch.zeros(1, *encoder_output.shape[1:], device=encoder_output.device),
                encoder_output,
            ]
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


class PointerNetwork(pl.LightningModule):

    END_SYMBOL_INDEX = 0

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
        # modules
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

    def reset_parameters(self) -> None:
        for param in self.parameters():
            nn.init.uniform_(param, *self.init_range)

    # NOTE: on ignore[override], pytorch_lightning's forward method
    # should not be typed, see torch.nn.Module, the method is just
    # not there (no need)
    def forward(
        self, encoder_input: PackedSequence, decoder_input: PackedSequence
    ) -> PackedSequence:
        # expand encoder init state to match LSTM signature
        batch_size = encoder_input.batch_sizes[0]
        encoder_init_state = (
            self.end_symbol.repeat(1, batch_size, 1),
            self.encoder_c_0.repeat(1, batch_size, 1),
        )
        encoder_output, encoder_last_state = self.encoder(
            encoder_input, encoder_init_state
        )
        # prepend start symbol to decoder input
        decoder_output, decoder_last_state = self.decoder(
            _prepend(decoder_input, self.start_symbol), encoder_last_state
        )
        # prepent end symbol to encoder output for attention
        scores: PackedSequence = self.attention(
            _prepend(encoder_output, self.end_symbol), decoder_output
        )
        return scores

    def training_step(self, batch: ptrnets.data._Batch, batch_idx: int) -> torch.Tensor:
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)
        # TODO: maybe append end symbol index inside dataset __getitem__
        # instead of here?
        target = _append(
            target, torch.tensor(self.END_SYMBOL_INDEX, device=target.data.device)
        )
        loss = self._get_loss(prediction, target)
        self.log("train/loss", loss.detach())
        self.log("train/token_acc", metrics.token_accuracy(prediction, target))
        self.log("train/sequence_acc", metrics.sequence_accuracy(prediction, target))
        return loss

    def _get_loss(self, prediction, target):
        return F.cross_entropy(prediction.data, target.data)

    def validation_step(
        self, batch: ptrnets.data._Batch, batch_idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # breakpoint()
        encoder_input, decoder_input, target = batch
        # TODO: same as previous todo
        target = _append(
            target, torch.tensor(self.END_SYMBOL_INDEX, device=target.data.device)
        )
        prediction = self(encoder_input, decoder_input)
        return prediction, target

    def validation_epoch_end(
        self, validation_step_outputs: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        all_predictions, all_targets = (
            _cat_packed_sequences(items) for items in zip(*validation_step_outputs)
        )

        self.log("val/loss", self._get_loss(all_predictions, all_targets))
        self.log("val/token_acc", metrics.token_accuracy(all_predictions, all_targets))
        self.log(
            "val/sequence_acc",
            metrics.sequence_accuracy(all_predictions, all_targets),
        )

    def test_step(
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

        breakpoint()

        return (
            encoder_input,
            nn.utils.rnn.pack_sequence(decoded, enforce_sorted=False),
            target,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learn_rate)


@dataclasses.dataclass
class _Beam:
    indices: tp.List[int]
    score: float
    decoder_input: torch.Tensor = dataclasses.field(repr=False)
    # see nn.LSTM docs for `last_hidden` type
    last_hidden: tp.Tuple[torch.Tensor, torch.Tensor] = dataclasses.field(repr=False)
    maxlen: tp.Optional[int] = None

    def is_done(self) -> bool:
        ends_with_zero = len(self.indices) > 0 and self.indices[-1] == 0
        is_too_long = len(self.indices) > self.maxlen if self.maxlen else False
        return is_too_long or ends_with_zero


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
                # go back to first node decoded and append 0 to indicate its done
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
            f"how am i supposed to return {nreturn} beams"
            f" when i am supposed to use only {k} beams!"
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


class PointerNetworkForConvexHull(PointerNetwork):
    @torch.no_grad()
    def decode(
        self,
        input: torch.Tensor,
        nbeams: int = 3,
        maxlen: tp.Optional[int] = None,
    ) -> tp.List[int]:
        assert input.ndim == 2, "input should be a 2 dim tensor, a sequence of points"
        maxlen = maxlen or input.shape[0] + 2
        encoder_output, encoder_last_hidden = self.encoder(
            nn.utils.rnn.pack_sequence([input])
        )
        encoder_output = _prepend_eos_token(encoder_output)

        beams: tp.List[_Beam] = [
            _Beam(
                indices=[],
                score=0.0,
                decoder_input=nn.utils.rnn.pack_sequence(
                    [torch.ones(1, 2, device=encoder_output.data.device) * -1]
                ),
                last_hidden=encoder_last_hidden,
                maxlen=maxlen,
            )
        ]

        while not all(beam.is_done() for beam in beams):
            # breakpoint()
            candidates: tp.List[_Beam] = []
            for beam in beams:
                if beam.is_done():
                    candidates.append(beam)
                    continue

                decoder_output, decoder_last_hidden = self.decoder(
                    beam.decoder_input, beam.last_hidden
                )
                # pad_packed_sequence returns a tuple, the padded
                # tensor and the original lengths, and then my padded
                # tensor has shape [1, 1, L] because its a batch of 1
                # and a single token for which to compute the attention
                # scores, so I select using [0][0, 0]
                attention_scores = nn.utils.rnn.pad_packed_sequence(
                    self.attention(encoder_output, decoder_output)
                )[0][0, 0]
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

        return beams[0].indices

    def test_epoch_end(
        self,
        test_step_outputs: tp.List[
            tp.Tuple[PackedSequence, PackedSequence, PackedSequence]
        ],
    ) -> None:
        all_point_sets, all_decoded, all_targets = (
            _cat_packed_sequences(items) for items in zip(*test_step_outputs)
        )
        poly_acc = metrics.polygon_accuracy(all_point_sets, all_decoded, all_targets)
        coverages = metrics.area_coverage(all_point_sets, all_decoded, all_targets)

        if (coverages < 0).sum() / coverages.shape[0] > 0.01:
            # this is considered as FAIL in the paper
            mean_coverage = -1.0
        else:
            mean_coverage = coverages[coverages >= 0].mean().item()
        self.log("poly_acc", poly_acc)
        self.log("mean_coverage", mean_coverage)
