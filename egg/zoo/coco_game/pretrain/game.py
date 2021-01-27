import torch


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
