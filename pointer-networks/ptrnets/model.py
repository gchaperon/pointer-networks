import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import pytorch_lightning as pl
import ptrnets
import functools
import typing as tp

import ptrnets.utils as utils


class Attention(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        # TODO: initialize weights properly
        # Naming convention follows the paper
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
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
            self.tanh(
                encoder_output @ self.W1 + (decoder_output @ self.W2).unsqueeze(1)
            )
            @ self.v
        )
        return self.softmax(scores).transpose(1, 2)

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
        # shape: (max_dec_seq_len, max_enc_sec_len, batch)
        scores = self.tanh(decoder_unpacked.unsqueeze(1) + encoder_unpacked) @ self.v
        # mask padded values with -inf so they cannot be attended to
        scores[
            :,
            torch.cat([torch.arange(l, scores.shape[1]) for l in encoder_lens]),
            torch.arange(scores.shape[2]).repeat_interleave(
                scores.shape[1] - encoder_lens
            ),
        ] = float("-inf")
        # shape: (max_dec_seq_len, max_enc_sec_len, batch)
        attention_coefs = self.softmax(scores)
        # breakpoint()
        return nn.utils.rnn.pack_padded_sequence(
            attention_coefs.transpose(1, 2), lengths=decoder_lens, enforce_sorted=False
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
        # breakpoint()
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
    ) -> tp.Tuple[PackedSequence, PackedSequence]:
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)
        return encoder_input, target, prediction

    def validation_epoch_end(
        self, validation_step_outputs: tp.List[tp.Tuple[PackedSequence, PackedSequence]]
    ) -> None:
        all_point_sets, all_targets, all_predictions = (
            torch.cat(list_, dim=1) for list_ in zip(*validation_step_outputs)
        )

        self.log("validation_loss", self._get_loss(all_predictions, all_targets))
        self.log(
            "validation_token_acc", utils.token_accuracy(all_predictions, all_targets)
        )
        self.log(
            "validation_sequence_acc",
            utils.sequence_accuracy(all_predictions, all_targets),
        )
        # use [:-1] to ignore the last index pointing to EOS token
        self.log(
            "validation_avg_dist_diff",
            utils.tour_distance(all_point_sets, all_predictions.argmax(2)[:-1]).mean()
            - utils.tour_distance(all_point_sets, all_targets[:-1]).mean(),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learn_rate)


class PointerNetworkForTSP(PointerNetwork):
    """Implements a single method, `decode` where logic for TSP is applied.
    This means, all points must be included in the results, and no point can appear
    twice in the response. Uses beam search plus these contraints"""

    @torch.no_grad()
    def decode(self, input: torch.Tensor):
        """This is super slow, maybe in the future (maybe) i will try to make
        it faster"""
        assert input.ndim == 2, "input should be a 2 dim tensor, a sequence of points"

        encoder_output, (h_n, c_n) = self.encoder(input.unsqueeze(1))
        encoder_output = self._prepend_bos_token(encoder_output)

        decoder_input = torch.ones(2) * -1
        indices = []
        while True:
            _, (h_n, c_n) = self.decoder(decoder_input.view(1, 1, -1), (h_n, c_n))
            attention_scores = self.attention(encoder_output, h_n)
            index = attention_scores.argmax(dim=2).item()
            breakpoint()
            if index == 0:
                break
            indices.append(index)
            decoder_input = input[index - 1]

        return indices
