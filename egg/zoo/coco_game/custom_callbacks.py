import json
from typing import List, Tuple

from egg.core import Callback, Interaction, ConsoleLogger
from egg.zoo.coco_game.utils.utils import console


class CustomEarlyStopperAccuracy(Callback):
    """
    A base class, supports the running statistic which is could be used for early stopping
    """

    def __init__(self, min_threshold: float, min_increase: float, field_name="accuracy") -> None:
        super(CustomEarlyStopperAccuracy, self).__init__()
        self.train_stats: List[Tuple[float, Interaction]] = []
        self.validation_stats: List[Tuple[float, Interaction]] = []
        self.epoch: int = 0

        self.max_threshold = min_threshold
        self.min_increase = min_increase
        self.val_field_name = field_name
        self.over_min_thr = False

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int) -> None:
        self.epoch = epoch
        self.train_stats.append((loss, logs))
        self.trainer.should_stop = self.should_stop(True)

    def on_test_end(self, loss: float, logs: Interaction, epoch: int) -> None:

        self.validation_stats.append((loss, logs))
        self.trainer.should_stop = self.should_stop(False)

    def should_stop(self, is_train: bool) -> bool:
        if not is_train:

            # must first increase past train thrs
            if not self.over_min_thr:
                return False

            if len(self.validation_stats) < 2:
                # wait at least two epochs for end
                return False

            loss, logs = self.validation_stats[-1]
            prev_loss, prev_logs = self.validation_stats[-2]

            mean = logs.aux[self.val_field_name].mean()
            prev_mean = prev_logs.aux[self.val_field_name].mean()

            if mean - prev_mean > self.min_increase:
                # if the increase is above the min thr, dont stop
                return False

            if loss - prev_loss < self.min_increase:
                return False

            console.log(f"Early stopping! Current val {mean}, prev val {prev_mean}")
            return True

        # train and not over min ths
        elif not self.over_min_thr:

            loss, logs = self.train_stats[-1]
            mean = logs.aux[self.val_field_name].mean()

            if mean > self.max_threshold:
                self.over_min_thr = True
