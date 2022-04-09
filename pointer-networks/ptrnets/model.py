import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import pytorch_lightning as pl
import ptrnets
import functools
import operator
import typing as tp
import ptrnets.utils as utils
import dataclasses


class Attention(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        # Naming convention follows the paper
        self.activation = nn.Tanh()
        self.W1 = nn.Parameter(torch.empty(input_size, input_size))
        self.W2 = nn.Parameter(torch.empty(input_size, input_size))
        self.v = nn.Parameter(torch.empty(input_size))

    @functools.singledispatchmethod
    def forward(self, arg: tp.Any, *args) -> tp.NoReturn:
        raise NotImplementedError(
            f"Forward not implemented for argument of type {type(arg)}"
        )

    @forward.register
    def _(
        self, encoder_output: torch.Tensor, decoder_output: torch.Tensor
    ) -> torch.Tensor:
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

    @forward.register
    def _(
        self,
        encoder_output: PackedSequence,
        decoder_output: PackedSequence,
    ) -> PackedSequence:
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
            self.activation(decoder_unpacked.unsqueeze(1) + encoder_unpacked) @ self.v
        )
        return nn.utils.rnn.pack_padded_sequence(
            scores.transpose(1, 2), lengths=decoder_lens, enforce_sorted=False
        )


@tp.overload
def _prepend_bos_token(encoder_output: PackedSequence) -> PackedSequence:
    ...


@tp.overload
def _prepend_bos_token(encoder_output: torch.Tensor) -> torch.Tensor:
    ...


def _prepend_bos_token(encoder_output):
    if isinstance(encoder_output, PackedSequence):
        # shape: (max_enc_seq_len, batch, hidden_size)
        encoder_unpacked, encoder_lens = nn.utils.rnn.pad_packed_sequence(
            encoder_output
        )
        encoder_unpacked = _pad_with_zeros(encoder_unpacked)
        return nn.utils.rnn.pack_padded_sequence(
            encoder_unpacked, lengths=encoder_lens + 1, enforce_sorted=False
        )
    else:
        return _pad_with_zeros(encoder_output)


def _pad_with_zeros(input: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, *input.shape[1:], device=input.device),
            input,
        ]
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

        self.encoder = nn.LSTM(  # type: ignore[no-untyped-call]
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=False,
        )
        self.decoder = nn.LSTM(  # type: ignore[no-untyped-call]
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

    @tp.overload
    def forward(
        self, encoder_input: PackedSequence, decoder_input: PackedSequence
    ) -> PackedSequence:
        ...

    @tp.overload
    def forward(
        self, encoder_input: torch.Tensor, decoder_input: torch.Tensor
    ) -> torch.Tensor:
        ...

    # NOTE: on ignore[override], pytorch_lightning's forward method
    # should not be typed, see torch.nn.Module, the method is just
    # not there (no need)
    def forward(self, encoder_input, decoder_input):  # type: ignore[override]
        encoder_output, encoder_last_state = self.encoder(encoder_input)
        decoder_output, decoder_last_state = self.decoder(
            decoder_input, encoder_last_state
        )
        # prepend vector of zeros to every example for eos token to point to
        encoder_output = _prepend_bos_token(encoder_output)
        return self.attention(encoder_output, decoder_output)

    def training_step(self, batch: ptrnets.data._Batch, batch_idx: int) -> torch.Tensor:
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)
        loss = self._get_loss(prediction, target)
        self.log("train_loss", loss)
        self.log("train_token_acc", utils.token_accuracy(prediction, target))
        self.log("train_sequence_acc", utils.sequence_accuracy(prediction, target))
        return loss

    def _get_loss(self, prediction, target):
        if isinstance(prediction, PackedSequence) and isinstance(
            target, PackedSequence
        ):
            return F.cross_entropy(prediction.data, target.data)
        else:
            return F.cross_entropy(
                prediction.flatten(start_dim=0, end_dim=1), target.flatten()
            )

    def validation_step(
        self, batch: ptrnets.data._Batch, batch_idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)
        return prediction, target

    def validation_epoch_end(
        self, validation_step_outputs: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        all_predictions, all_targets = (
            torch.cat(items, dim=1) for items in zip(*validation_step_outputs)
        )

        self.log("validation_loss", self._get_loss(all_predictions, all_targets))
        self.log(
            "validation_token_acc", utils.token_accuracy(all_predictions, all_targets)
        )
        self.log(
            "validation_sequence_acc",
            utils.sequence_accuracy(all_predictions, all_targets),
        )

    def test_step(
        self, batch: ptrnets.data._Batch, batch_idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_input, decoder_input, target = batch
        decoded = torch.stack(
            [
                torch.tensor(self.decode(points), device=target.device)
                for points in encoder_input.unbind(1)
            ],
        )
        target = target[:-1]
        return encoder_input, decoded.T, target

    def test_epoch_end(
        self,
        test_step_outputs: tp.List[tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        all_point_sets, all_decoded, all_targets = (
            torch.cat(items, dim=1) for items in zip(*test_step_outputs)
        )
        self.log(
            "sequence_acc(decodedd)",
            (all_decoded == all_targets).all(dim=0).sum() / all_decoded.shape[1],
        )
        target_avg_tour_distance = torch.mean(
            utils.tour_distance(all_point_sets, all_targets)
        )
        decoded_avg_tour_distance = torch.mean(
            utils.tour_distance(all_point_sets, all_decoded)
        )
        self.log(
            "avg_tour_distance_diff",
            decoded_avg_tour_distance - target_avg_tour_distance,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learn_rate)


@dataclasses.dataclass
class _Beam:
    indices: tp.List[int]
    score: float
    decoder_input: torch.Tensor
    # see nn.LSTM docs for `last_hidden` type
    last_hidden: tp.Tuple[torch.Tensor, torch.Tensor]

    def is_done(self):
        return self.indices and self.indices[-1] == 0


class PointerNetworkForTSP(PointerNetwork):
    """Implements a single method, `decode` where logic for TSP is applied.
    This means, all points must be included in the results, and no point can appear
    twice in the response. Uses beam search plus these contraints"""

    @torch.no_grad()
    def decode(
        self, input: torch.Tensor, k: int = 3, nreturn: int = 1, wscores: bool = False
    ) -> tp.Union[
        tp.Union[tp.List[int], tp.Tuple[tp.List[int], float]],
        tp.Union[tp.List[tp.List[int]], tp.List[tp.Tuple[tp.List[int], float]]],
    ]:
        """This is super slow, maybe in the future (maybe) i will try to make
        it faster"""
        assert input.ndim == 2, "input should be a 2 dim tensor, a sequence of points"
        nreturn = nreturn or k
        assert nreturn <= k, (
            f"how am i supposed to return {nreturn} beams"
            f"when i am supposed to use only {k} beams!"
            "\nTHINK MARK, THINK!"
        )

        encoder_output, encoder_last_hidden = self.encoder(input.unsqueeze(1))
        encoder_output = _prepend_bos_token(encoder_output)

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
                # go back to node 1 and append 0 to indicate its done
                if set(beam.indices) == set(range(1, len(input) + 1)):
                    candidates.append(
                        dataclasses.replace(beam, indices=beam.indices + [1, 0])
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

                for prob, index in zip(probs[:k], indices[:k]):
                    candidates.append(
                        _Beam(
                            indices=[*beam.indices, index.item()],
                            score=beam.score - torch.log(prob).item(),
                            decoder_input=input[index - 1],
                            last_hidden=(h_n, c_n),
                        )
                    )
            beams = sorted(candidates, key=operator.attrgetter("score"))[:k]

        if nreturn == 1:
            if wscores:
                return beams[0].indices[:-1], beams[0].score
            else:
                return beams[0].indices[:-1]
        else:
            if wscores:
                return [(beam.indices[:-1], beam_score) for beam in beams[:nreturn]]
            else:
                return [beam.indices[:-1] for beam in beams[:nreturn]]
