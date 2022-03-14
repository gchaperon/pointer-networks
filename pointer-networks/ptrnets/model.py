import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import pytorch_lightning as pl
import ptrnets
import typing as tp


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

    def forward(
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

    # NOTE: on ignore[override], pytorch_lightning's forward method
    # should not be typed, see torch.nn.Module, the method is just
    # not there (no need)
    def forward(  # type: ignore[override]
        self,
        encoder_input: PackedSequence,
        decoder_input: PackedSequence,
    ) -> PackedSequence:
        # breakpoint()
        encoder_output, encoder_last_state = self.encoder(encoder_input)
        decoder_output, decoder_last_state = self.decoder(
            decoder_input, encoder_last_state
        )
        # prepend vector of zeros to every example for eos token to point to
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
        encoder_output = nn.utils.rnn.pack_padded_sequence(
            encoder_unpacked, lengths=encoder_lens + 1, enforce_sorted=False
        )
        return self.attention(encoder_output, decoder_output)

    def training_step(self, batch: ptrnets.data._Batch, batch_idx: int) -> torch.Tensor:
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)
        # breakpoint()
        loss = F.cross_entropy(prediction.data, target.data)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: ptrnets.data._Batch, batch_idx: int
    ) -> tp.Tuple[PackedSequence, PackedSequence]:
        encoder_input, decoder_input, target = batch
        prediction = self(encoder_input, decoder_input)
        return target, prediction

    def validation_epoch_end(
        self, validation_step_outputs: tp.List[tp.Tuple[PackedSequence, PackedSequence]]
    ) -> None:
        all_targets: tp.List[PackedSequence]
        all_predictions: tp.List[PackedSequence]
        all_targets, all_predictions = zip(*validation_step_outputs)
        loss = F.cross_entropy(
            torch.cat([t.data for t in all_predictions]),
            torch.cat([t.data for t in all_targets]),
        )
        self.log("validation_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learn_rate)
