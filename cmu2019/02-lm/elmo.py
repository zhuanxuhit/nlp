import torch
from torch import nn
import logging
from onmt.encoders import RNNEncoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Elmo(torch.nn.Module):

    def __init__(self, hidden_size, dropout, embeddings: nn.Embedding):
        super(Elmo, self).__init__()
        logger.info("Initializing ELmo")
        self.lstm = RNNEncoder(
            rnn_type="LSTM",
            bidirectional=True,
            num_layers=2,
            hidden_size=hidden_size,
            dropout=dropout,
            embeddings=embeddings
        )

    def forward(self, ):
        pass
