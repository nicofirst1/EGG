from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class FakeInteraction:
    # incoming data
    sender_input: Optional
    receiver_input: Optional
    labels: Optional

    # what agents produce
    message: Optional
    receiver_output: Optional

    # auxilary info
    message_length: Optional
    aux: Dict

    def to(self, *args, **kwargs) -> "Interaction":
        """Moves all stored tensor to a device. For instance, it might be not
        useful to store the interaction logs in CUDA memory."""

        def _to(x):
            if x is None or not torch.is_tensor(x):
                return x
            return x.to(*args, **kwargs)

        self.sender_input = _to(self.sender_input)
        self.receiver_input = _to(self.receiver_input)
        self.labels = _to(self.labels)
        self.message = _to(self.message)
        self.receiver_output = _to(self.receiver_output)
        self.message_length = _to(self.message_length)

        if self.aux:
            self.aux = dict((k, _to(v)) for k, v in self.aux.items())

        return self


class PretrainGame(torch.nn.Module):
    def __init__(self, sender, loss, opts, train_strategy, test_strategy):
        super().__init__()
        self.sender = sender
        self.criterion = loss
        self.opts = opts
        self.train_logging_strategy = train_strategy
        self.test_logging_strategy = test_strategy

    def forward(self, sender_input, labels, receiver_input=None):
        outputs = self.sender(sender_input)
        loss, metrics = self.criterion(sender_input, None, None, outputs, labels)
        loss = loss.sum()

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            message=torch.rand((self.opts.batch_size, self.opts.max_len)),
            receiver_output=outputs.detach(),
            message_length=torch.rand((self.opts.batch_size, 1)),
            aux=metrics,
        )

        return loss, interaction
