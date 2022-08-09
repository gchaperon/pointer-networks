import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import pytorch_lightning as pl
import ptrnets
import typing as tp
import torchmetrics
import ptrnets.metrics as metrics


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


def _unravel_index(
    indices: torch.Tensor, shape: tp.Tuple[int, ...]
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    # Supper innefficient to copy to cpu and then back to cuda if indices
    # is a cuda tensor, but for now it suffices.
    device = indices.device
    unraveled_coords = np.unravel_index(indices.cpu().numpy(), shape)
    return tuple(torch.tensor(arr, device=device) for arr in unraveled_coords)


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
        self.train_metrics = metric_collection.clone(prefix="train/")
        self.val_metrics = metric_collection.clone(prefix="val/")
        self.test_metrics = torchmetrics.MetricCollection(
            {
                "polygon_accuracy": metrics.PolygonAccuracy(),
                "polygon_coverage": metrics.AverageAreaCoverage(),
            },
            prefix="test/",
        )

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

    def validation_step(self, batch: ptrnets.data._Batch, batch_idx: int) -> None:
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
        encoder_input, _, target = batch
        decoded = self.decode(encoder_input)
        self.log_dict(
            self.test_metrics(encoder_input, decoded, target),
            batch_size=target.batch_sizes[0],
        )

    def configure_optimizers(self) -> tp.Dict["str", tp.Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learn_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.96
                ),
                "interval": "epoch",
                "name": "lr/adam",
            },
        }

    def decode(self, encoder_input: PackedSequence, nbeams: int = 0) -> PackedSequence:
        raise NotImplementedError("`decode` should be implemented by subclasses")


class PointerNetworkForConvexHull(PointerNetwork):
    @torch.no_grad()
    def batch_beam_search(
        self, encoder_input: PackedSequence, nbeams: int = 3
    ) -> PackedSequence:
        """Decodes a sequence batch using `nbeams` beams. All elements of the
        batch and beams of each sequence are decoded in parallel"""

        # +2 because solution has to loop and then add 0
        max_len = len(encoder_input.batch_sizes) + 2
        batch_size: int = encoder_input.batch_sizes[0].item()

        # repeat inputs
        input_padded, input_lens = nn.utils.rnn.pad_packed_sequence(encoder_input)
        input_padded = torch.repeat_interleave(input_padded, nbeams, dim=1)
        input_lens = torch.repeat_interleave(input_lens, nbeams, dim=0)
        encoder_input = nn.utils.rnn.pack_padded_sequence(
            input_padded, input_lens, enforce_sorted=False
        )

        # initial state
        encoder_output, last_hidden = self._encoder_forward(encoder_input)
        decoder_input = self.start_symbol.repeat(1, batch_size * nbeams, 1)
        beams = torch.empty(0, batch_size * nbeams, dtype=int, device=self.device)
        beam_scores = torch.tensor(
            [0.0, *[torch.inf] * (nbeams - 1)] * batch_size, device=self.device
        )

        for i in range(max_len):
            _, last_hidden = self.decoder(decoder_input, last_hidden)
            logits = nn.utils.rnn.pad_packed_sequence(
                self.attention(encoder_output, last_hidden[0])
            )[0].squeeze(0)
            # There are some conditions that a sequence must satisfy
            # * the first three tokens must be different and nonzero, else the
            #   sequence is not a polygon
            # * tokens in a sequence must be unique, except for the first one
            #   which is used to close the polygon
            # * the token 0 can only be produced after the polygon has been
            #   closed
            # * once a 0 has been produced, i.e. the beam is finished, only
            #   zeros can be predicted with the highest probability
            #
            # For this function I will only enforce the first condition, the
            # fact that no intermediate point can be repeated and the fact that
            # finished beams predict only zero.  I will let the net predictt 0
            # at any point, i.e. the net can predict "open" sequences, that
            # will be closed implicitly by the function computing polygon
            # metrics.
            if i < 3:
                # shouldn't predict neither 0 nor any of the already seen values
                logits[:, 0] = -torch.inf
                logits[torch.arange(batch_size * nbeams), beams] = -torch.inf
            else:
                # is finished if has a zero anywhere or the first element is
                # repeated, ie the polygon has closed
                #
                # finished should only predict 0, unfinished should only
                # predict new values or the first that was predicted in the
                # beam
                finished_mask = torch.any(beams == 0, dim=0) | torch.any(
                    beams[1:] == beams[0], dim=0
                )

                logits[finished_mask, 1:] = -torch.inf
                logits[~finished_mask, beams[1:, ~finished_mask]] = -torch.inf

            probs = logits.softmax(1)
            temp_scores = torch.log(probs)
            # replace -inf with super negative value, but not -inf
            temp_scores[temp_scores == -torch.inf] = torch.finfo().min
            new_beam_scores = beam_scores[:, None] - temp_scores
            topk_scores, indices = torch.topk(
                new_beam_scores.view(batch_size, -1),
                k=nbeams,
                largest=False,
                sorted=True,
            )
            # now both of shape (batch_size, nbeams)
            beams_index, index_prediction = _unravel_index(
                indices, shape=(nbeams, new_beam_scores.shape[1])
            )

            # flatten beams_index and index_prediction
            beams_index = beams_index.view(-1) + (
                torch.arange(batch_size, device=self.device) * nbeams
            )[:, None].repeat(1, nbeams).view(-1)
            index_prediction = index_prediction.view(-1)
            beams = torch.vstack([beams[:, beams_index], index_prediction])
            beam_scores = topk_scores.view(-1)

            # update recurrent state
            # choose the last hidden from the correct beams
            last_hidden = (
                last_hidden[0][:, beams_index],
                last_hidden[1][:, beams_index],
            )
            assert torch.all(
                index_prediction.cpu() < input_lens + 1
            ), "some predictions are out of bounds"
            decoder_input = input_padded[
                None, index_prediction - 1, torch.arange(batch_size * nbeams)
            ]
            # if all finished, stop decoding
            if torch.all(torch.any(beams == 0, dim=0)):
                break
        else:
            # if no break occurs, add last 0 to al predictions
            beams = torch.vstack(
                [beams, torch.zeros(batch_size, nbeams).type_as(beams)]
            )

        # reshape beam_scores and beams to (batch_size, nbeams)
        beam_scores = beam_scores.view(batch_size, nbeams)
        beams = beams.view(-1, batch_size, nbeams)

        assert torch.all(beam_scores.argmin(dim=1) == 0), (
            "all best beam scores should be in possition 0, since topk"
            " is sorted by default, but they weren't :-/"
        )
        winners = beams[..., 0]
        return nn.utils.rnn.pack_padded_sequence(
            winners, lengths=winners.argmin(dim=0).cpu() + 1, enforce_sorted=False
        )

    decode = batch_beam_search
