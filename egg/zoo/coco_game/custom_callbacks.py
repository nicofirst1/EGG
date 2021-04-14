from os.path import join
from pathlib import Path
from typing import List, Tuple

import torch
from pycocotools.coco import COCO
from torch.utils.tensorboard import SummaryWriter

from egg.core import Callback, Interaction, LoggingStrategy
from egg.zoo.coco_game.utils.utils import console, get_labels


class SyncLogging(LoggingStrategy):
    """
    Log strategy based on random probability
    """

    def __init__(self, logging_step: int, *args):
        self.logging_step = logging_step
        self.cur_batch = 0
        super().__init__(*args)

    def filtered_interaction(
            self,
            sender_input,
            receiver_input,
            labels,
            message,
            receiver_output,
            message_length,
            aux,
    ):
        should_store = False
        if self.cur_batch % self.logging_step == 0:
            should_store = True

        self.cur_batch += 1

        return Interaction(
            sender_input=sender_input if should_store else None,
            receiver_input=receiver_input if should_store else None,
            labels=labels if should_store else None,
            message=message if should_store else None,
            receiver_output=receiver_output if should_store else None,
            message_length=message_length if should_store else None,
            aux=aux,
        )


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
            "Pred SuperClass",
            "True Class",
            "True SuperClass",
            "Is correct",
            "Distractor class",
            "Distractor SuperClass",
            "Other Classes",
            "Other SuperClasses",
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

    def get_data(self, objects, class_ids, predictions, true_seg):

        true_class = []
        distractors = []

        for idx in range(len(objects)):
            tc = objects[idx].pop(true_seg[idx])
            true_class.append(tc)
            distractors.append(objects[idx])

        predictions = torch.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        correct_pred = predictions == true_seg

        predictions = (
            class_ids
                .gather(1, predictions.unsqueeze(dim=1))
                .squeeze()
                .tolist()
        )

        res_dict = dict(
            predictions=predictions,
            true_class=true_class,
            distractors=distractors,
            correct_pred=correct_pred,
        )

        return res_dict

    def log_interactions_file(self, logs: Interaction, global_step: int):
        """
        Logs the csv with message
        """





        def get_annotation_id(annotations):
            return [x['id'] for x in annotations]

        def get_annotation_cat(ann_ids):
            return [self.val_coco.anns[idx]["category_id"] for idx in ann_ids]

        def get_other_cats(image_id, used_annotations):
            other_ans = [self.val_coco.imgToAnns[x].copy() for x in image_id.tolist()]
            other_ans = [get_annotation_id(annotations) for annotations in other_ans]

            # remove distractors/target
            for idx in range(len(other_ans)):
                oa = other_ans[idx]
                ua = used_annotations[idx].tolist()
                for x in ua:
                    oa.remove(x)
                other_ans[idx] = oa

            other_ans = [get_annotation_cat(x) for x in other_ans]

            return other_ans

        if self.val_coco is None:
            return

        # extract data from logs
        res_dict = get_labels(logs.labels)
        true_seg = res_dict["target_position"]
        objects = res_dict["class_id"]
        objects = [x.tolist() for x in objects]
        image_id = res_dict["image_id"]
        messages = logs.message

        # filter data
        data_dict = self.get_data(objects, res_dict['class_id'], logs.receiver_output, true_seg)

        predictions = data_dict['predictions']
        true_class = data_dict['true_class']
        distractors = data_dict['distractors']
        correct_pred = data_dict['correct_pred']

        # define lambdas
        get_cat_name_id = lambda cat_ids: [self.val_coco.cats[idx]["name"] for idx in cat_ids]
        get_supercat_name_id = lambda cat_ids: [self.val_coco.cats[idx]["supercategory"] for idx in cat_ids]

        # Getting superclasses
        pred_super_cats = get_supercat_name_id(predictions)
        true_superclass = get_supercat_name_id(true_class)
        dist_super_cat = [get_supercat_name_id(idx) for idx in distractors]



        # Getting classes
        pred_cats = get_cat_name_id(predictions)
        true_class = get_cat_name_id(true_class)
        dist_cat = [get_cat_name_id(idx) for idx in distractors]

        # get all other objects in image
        other_ans = get_other_cats(image_id, res_dict['ann_id'])

        other_supercats = [get_supercat_name_id(idx) for idx in other_ans]
        other_cats = [get_cat_name_id(idx) for idx in other_ans]

        with open(self.message_file, "a+") as file:
            for idx in range(len(true_class)):
                # 'Epoch,Message,Pred Class,Pred SuperClass,True Class,True SuperClass,Is correct,Distractor class,Distractor SuperClass,Other Classes,Other SuperClasses'

                line = f"{global_step},"
                line += f"{';'.join([str(x) for x in messages[idx].tolist()])},"
                line += f"{pred_cats[idx]},"
                line += f"{pred_super_cats[idx]},"
                line += f"{true_class[idx]},"
                line += f"{true_superclass[idx]},"
                line += f"{correct_pred[idx]},"
                line += f"{';'.join(dist_cat[idx])},"
                line += f"{';'.join(dist_super_cat[idx])},"
                line += f"{';'.join(other_cats[idx])},"
                line += f"{';'.join(other_supercats[idx])}"
                line += "\n"
                file.write(line)
