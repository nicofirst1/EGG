import json
from os.path import join
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from pycocotools.coco import COCO
from torch.utils.tensorboard import SummaryWriter

from egg.core import Callback, Interaction, ConsoleLogger, LoggingStrategy
from egg.zoo.coco_game.utils.utils import console, get_labels


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


class InteractionCSV(Callback):
    def __init__(
            self,
            tensorboard_dir: str,
            val_coco: COCO = None,
    ):
        """
        Callback to log metrics to tensorboard

        """
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.message_file = Path(join(tensorboard_dir, "interactions.csv"))
        self.val_coco = val_coco
        self.init_message_file()

    def init_message_file(self):
        header = [
            "Epoch",
            "Message",
            "Pred Class",
            "True Class",
            "Is correct",
            "Distractors",
            "Other Classes",
        ]

        if self.message_file.exists():
            console.log(
                f"File {self.message_file} already exists, data will be appended"
            )
        else:
            with open(self.message_file, "w+") as f:
                f.write(",".join(header))
                f.write("\n")

    def on_train_end(self):
        self.writer.close()

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        # self.log_precision_recall(logs, phase="val", global_step=epoch)
        self.log_interactions_file(logs, global_step=epoch)

    def log_interactions_file(self, logs: Interaction, global_step: int):
        """
        Logs the csv with message
        """

        def get_cat_name_id(cat_ids):
            return [self.val_coco.cats[idx]["name"] for idx in cat_ids]

        def get_cat_name_ann(annotations):
            """
            Return list of objs names from annotation ids
            """

            result = []

            for ann in annotations:
                ann_id = ann["category_id"]
                ann_name = self.val_coco.cats[ann_id]["name"]
                result.append(ann_name)
            return result

        if self.val_coco is None:
            return

        res_dict = get_labels(logs.labels)
        true_seg = res_dict["true_segment"]
        objects = res_dict["class_id"]
        objects = [x.tolist() for x in objects]

        true_class = []
        distractors = []

        for idx in range(len(objects)):
            tc = objects[idx].pop(true_seg[idx])
            true_class.append(tc)
            distractors.append(objects[idx])

        image_id = res_dict["image_id"]
        messages = logs.message
        predictions = logs.receiver_output
        predictions = torch.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        correct_pred = predictions == true_seg

        predictions = (
            res_dict["class_id"]
                .gather(1, predictions.unsqueeze(dim=1))
                .squeeze()
                .tolist()
        )
        predictions = get_cat_name_id(predictions)

        true_class = [self.val_coco.cats[idx]["name"] for idx in true_class]
        distractors = [get_cat_name_id(idx) for idx in distractors]

        # get all other objects in image
        other_ans = [self.val_coco.imgToAnns[x].copy() for x in image_id.tolist()]

        # transfrom id to string
        other_ans = [get_cat_name_ann(x) for x in other_ans]

        # remove obj to predict from other anns
        for idx in range(len(true_class)):
            oa = other_ans[idx]
            tc = true_class[idx]
            oa.remove(tc)
            dis = distractors[idx]
            for d in dis:
                if d in oa:
                    oa.remove(d)

        with open(self.message_file, "a+") as file:
            for idx in range(len(true_class)):
                # ["Epoch", "Message", "Pred Class", "True Class","Is correct", Distractors, "Other Classes"]

                line = f"{global_step},"
                line += f"{';'.join([str(x) for x in messages[idx].tolist()])},"
                line += f"{predictions[idx]},"
                line += f"{true_class[idx]},"
                line += f"{correct_pred[idx]},"
                line += f"{';'.join(distractors[idx])},"
                line += f"{';'.join(other_ans[idx])}\n"
                file.write(line)

