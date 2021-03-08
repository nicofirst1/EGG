from typing import List, Tuple

from egg.core import Callback, Interaction
from egg.zoo.coco_game.utils.utils import console


class RlScheduler(Callback):
    def __init__(self, rl_optimizer):
        self.rl_optimizer = rl_optimizer

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.rl_optimizer.step()


class EarlyStopperAccuracy(Callback):
    """
    A base class, supports the running statistic which is could be used for early stopping
    """

    def __init__(self, max_threshold: float, min_increase: float) -> None:
        super(EarlyStopperAccuracy, self).__init__()
        self.train_stats: List[Tuple[float, Interaction]] = []
        self.validation_stats: List[Tuple[float, Interaction]] = []
        self.epoch: int = 0

        self.max_threshold = max_threshold
        self.min_increase = min_increase
        self.val_field_name = "accuracy_receiver"
        self.under_max_thr = False

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
            if not self.under_max_thr:
                return False

            if len(self.validation_stats) < 2:
                # wait at least two epochs for end
                return False

            loss, logs = self.validation_stats[-1]
            loss, prev_logs = self.validation_stats[-2]

            mean = logs.aux[self.val_field_name].mean()
            prev_mean = prev_logs.aux[self.val_field_name].mean()

            if mean - prev_mean > self.min_increase:
                # if the increase is above the min thr, dont stop
                return False

            console.log(f"Early stopping! Current val {mean}, prev val {prev_mean}")
            return True

        # train and not over min ths
        elif not self.under_max_thr:

            loss, logs = self.train_stats[-1]

            if loss < self.max_threshold:
                self.under_max_thr = True
