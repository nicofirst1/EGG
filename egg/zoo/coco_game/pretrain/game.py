import pathlib
import sys
from typing import Union

import torch

from egg.core.callbacks import Checkpoint, CheckpointSaver
from egg.zoo.coco_game.losses import Losses


class PretrainGame(torch.nn.Module):
    def __init__(self, sender, loss: Losses.final_loss, opts, train_strategy, val_strategy):
        super().__init__()
        self.sender = sender
        self.criterion = loss
        self.opts = opts
        self.train_logging_strategy = train_strategy
        self.val_logging_strategy = val_strategy

    def forward(self, sender_input, labels, receiver_input=None):
        outputs = self.sender(sender_input)

        loss, metrics = self.criterion(sender_input=sender_input, message=None, receiver_input=None,
                                       receiver_output=None, labels=labels, sender_output=outputs)
        loss = loss.mean()

        logging_strategy = (
            self.train_logging_strategy if self.training else self.val_logging_strategy
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


class SenderSaver(CheckpointSaver):
    def __init__(
            self,
            sender: torch.nn.Module,
            checkpoint_path: Union[str, pathlib.Path],
            checkpoint_freq: int = 1,
            prefix: str = "",
            max_checkpoints: int = sys.maxsize,
    ):
        """Saves a checkpoint file for training.
        :param checkpoint_path:  path to checkpoint directory, will be created if not present
        :param checkpoint_freq:  Number of epochs for checkpoint saving
        :param prefix: Name of checkpoint file, will be {prefix}{current_epoch}.tar
        :param max_checkpoints: Max number of concurrent checkpoint files in the directory.
        """
        super(SenderSaver, self).__init__(checkpoint_path, checkpoint_freq, prefix, max_checkpoints)
        self.sender = sender

    def get_checkpoint(self):
        return Checkpoint(
            epoch=self.epoch_counter,
            model_state_dict=self.sender.state_dict(),
            optimizer_state_dict={},
        )
