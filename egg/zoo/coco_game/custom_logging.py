import copy
import random
from os.path import join
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from egg.core import Callback, Interaction, LoggingStrategy
from egg.zoo.coco_game.utils.utils import console, get_labels, get_true_elems


class RandomLogging(LoggingStrategy):
    """
    Log strategy based on random probability
    """

    def __init__(self, logging_step: int, store_prob: float = 1, *args):
        self.store_prob = store_prob
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
        rnd = random.random()
        should_store = rnd < self.store_prob
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


class TensorboardLogger(Callback):
    def __init__(
            self,
            tensorboard_dir: str,
            loggers: Dict[str, LoggingStrategy] = None,
            train_logging_step: int = 50,
            val_logging_step: int = 20,
            resume_training: bool = False,
            game: torch.nn.Module = None,
            class_map: Dict[int, str] = {},
            get_image_method=None,
            hparams=None,
    ):
        """
        Callback to log metrics to tensorboard
        Args:
            tensorboard_dir: path to where to write metrics
            loggers: Dictionary containing the RandomLoggingStrategy, use to update the current batch value
            train_logging_step: number of batches to log in training
            val_logging_step:  number of batches to log in validation
            resume_training: if the train is resumed, load previous global steps
            game: A torch module used to log the graph
        """
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.gs_file = join(tensorboard_dir, "gs.txt")
        self.train_log_step = train_logging_step
        self.val_log_step = val_logging_step
        self.train_gs = 0
        self.val_gs = 0
        self.loggers = loggers

        self.game = game
        self.class_map = class_map
        self.get_images = get_image_method

        self.embeddings_log_step = 4
        self.log_conv = False
        self.log_graph = False
        self.embedding_num = 300

        self.hparam = self.filter_hparam(hparams)

        if resume_training:
            try:
                self.get_gs()
            except FileNotFoundError:
                pass


    @staticmethod
    def filter_hparam(hparam):
        allowed_types = [int, float, str, bool, torch.Tensor]
        hparam = {
            k: v
            for k, v in hparam.items()
            if any([isinstance(v, t) for t in allowed_types])
        }
        return hparam

    def get_gs(self):
        """
        Read global steps from file
        """
        with open(self.gs_file, "r+") as f:
            line = f.read()

        self.train_gs = int(line.split(",")[0])
        self.val_gs = int(line.split(",")[1])

    def save_gs(self):
        """
        Dump global steps from file
        """
        with open(self.gs_file, "w+") as f:
            f.write(f"{self.train_gs},{self.val_gs}")

    def on_train_end(self):
        self.writer.close()

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):

        # self.log_precision_recall(logs, phase="train", global_step=epoch)
        self.log_messages_embedding(logs, is_train=True, global_step=epoch)
        if self.log_conv:
            self.log_conv_filter(logs, phase="train", global_step=epoch)
        if self.log_graph:
            self.log_graphs(logs)
            self.log_graph = False

        self.writer.add_scalar("epoch", epoch, global_step=self.train_gs)
        if self.loggers is not None:
            self.loggers["train"].cur_batch = 0

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        # self.log_precision_recall(logs, phase="val", global_step=epoch)
        self.log_messages_embedding(logs, is_train=False, global_step=epoch)
        self.log_hparams(logs, loss)
        if self.log_conv:
            self.log_conv_filter(logs, phase="train", global_step=epoch)
        if self.loggers is not None:
            self.loggers["val"].cur_batch = 0

        self.log_conv = False

    def on_batch_end(
            self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):

        if batch_id != 0:
            if batch_id % self.train_log_step == 0 and is_training:
                self.log(loss.detach(), logs, is_training)
            if batch_id % self.val_log_step == 0 and not is_training:
                self.log(loss.detach(), logs, is_training)

    def log_receiver_output(
            self, receiver_output: torch.Tensor, phase: str, global_step: int
    ):
        """
        Add information about receiver output to tensorboard.
        Precisely adds:
         - the bboxs as an image
         - the bbox 4 coordinates as histograms
         - pred classes as histogram
        """

        # add pred class histograms
        pred_classes = torch.argmax(receiver_output, dim=-1)
        num_classes = receiver_output.shape[1]

        self.writer.add_histogram(
            tag=f"{phase}/pred_class",
            values=pred_classes,
            global_step=global_step,
            bins=num_classes,
        )

    def log_metrics(
            self,
            logs: Interaction,
            phase: str,
            global_step: int,
            loss: float,
    ):

        metrics = logs.aux

        self.writer.add_scalar(
            tag=f"{phase}/loss", scalar_value=loss, global_step=global_step
        )

        metrics = {k: v.mean() for k, v in metrics.items()}
        for k, v in metrics.items():
            self.writer.add_scalar(
                tag=f"{phase}/{k}", scalar_value=v, global_step=global_step
            )

    def log_graphs(self, logs: Interaction, use_sender=False):
        """
        Logs the game graph once if not None.
        Game can either be the sender or the receiver in a game

        """
        device = next(self.game.parameters()).device

        if use_sender:
            inp = logs.sender_input.to(device)
            model = self.game.sender
        else:
            inp = [
                logs.message.to(device),
                logs.receiver_input.to(device),
            ]
            model = self.game.receiver

        self.writer.add_graph(model=model, input_to_model=inp)

    def log_hparams(self, logs: Interaction, loss: float):
        """
        Logs Hparam with metrics
        """
        if self.hparam is None:
            return
        metrics = logs.aux
        metrics = {k: v.mean() for k, v in metrics.items()}
        metrics["loss"] = loss
        metrics = {f"hparams/{k}": v for k, v in metrics.items()}

        self.writer.add_hparams(
            hparam_dict=self.hparam, metric_dict=metrics, run_name="hparams"
        )

    def log_labels(
            self, logs: Interaction, phase: str, global_step: int, label_key="true_segment"
    ):
        """
        Logs statistic about the labels such as the class and the bounding boxes
        """
        labels = logs.labels

        # classes
        res_dict = get_labels(labels)
        true_classes = res_dict[label_key]

        self.writer.add_histogram(
            tag=f"{phase}/true_class",
            values=true_classes,
            global_step=global_step,
        )

    def log_conv_filter(self, logs: Interaction, phase: str, global_step: int):

        pretrained = self.game.sender.agent.vision

        # port to cpu
        pretrained = copy.deepcopy(pretrained).cpu()

        # get info
        batch_size = logs.labels.shape[0]
        idx = random.randint(0, batch_size)

        # get random images
        sender_img = logs.sender_input[idx]

        # split img/seg
        img_size = sender_img.shape[1]
        sender_img = sender_img[:, :, :img_size]

        # add batch size dimension
        sender_img = sender_img.unsqueeze(dim=0)

        activation = {}

        # function for hook
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        # register all conv layers for hook
        for i in range(len(pretrained)):
            if type(pretrained[i]) == torch.nn.Conv2d:
                pretrained[i].register_forward_hook(get_activation(f"conv{i}"))
            elif type(pretrained[i]) == torch.nn.Sequential:
                for j in range(len(pretrained[i])):
                    for child in pretrained[i][j].children():
                        if type(child) == torch.nn.Conv2d:
                            pretrained[i][j].register_forward_hook(
                                get_activation(f"conv{i + j}")
                            )

        # run trough pretrained
        pretrained(sender_img)

        for k, v in activation.items():
            # permute to have [filters, 1, H,W]
            tensor = v.permute((1, 0, 2, 3))
            # get only first 64 filters
            tensor = tensor[:64, :, :, :]
            grid = make_grid(tensor, normalize=True)

            self.writer.add_image(
                tag=f"{phase}/{k}/sender_img",
                img_tensor=grid,
                global_step=global_step,
            )

    def log_precision_recall(self, logs: Interaction, phase: str, global_step: int):
        """
        Log precision recall for every class
        """

        # extract classification info
        res_dict = get_labels(logs.labels)
        true_classes = res_dict["class_id"]
        predictions = logs.receiver_output
        predictions = torch.softmax(predictions, dim=1)

        num_classes = predictions.shape[1]

        for idx in range(num_classes):
            tensorboard_preds = true_classes == idx
            tensorboard_probs = predictions[:, idx]

            if idx in self.class_map.keys():
                class_name = self.class_map[idx]
            else:
                class_name = str(idx)

            class_name = f"{phase}/{class_name}"

            self.writer.add_pr_curve(
                class_name,
                tensorboard_preds,
                tensorboard_probs,
                global_step=global_step,
            )

    def log_messages_embedding(
            self, logs: Interaction, is_train: bool, global_step: int
    ):
        """
        Logs the messages as an embedding
        """

        image_size = (200, 200)

        if global_step % self.embeddings_log_step != 0:
            return

        res_dict = get_labels(logs.labels)
        true_seg = res_dict["true_segment"]

        true_class, ann_id = get_true_elems(
            true_seg, res_dict["class_id"], res_dict["ann_id"]
        )
        true_class = [x.tolist() for x in true_class]
        true_class = np.asarray(true_class)
        ann_id = [x.tolist() for x in ann_id]
        ann_id = np.asarray(ann_id)
        image_id = res_dict["image_id"]

        if self.get_images is not None:
            # sample down the number of images to load to 200
            to_log = random.sample(
                range(len(true_class)),
                k=min(self.embedding_num, len(true_class)),
            )
            image_id = image_id[to_log]
            true_class = true_class[to_log]
            ann_id = ann_id[to_log]
            messages = logs.message[to_log]

            if is_train:
                imgs = self.get_images(image_id.tolist(), ann_id, True, image_size)
            else:
                imgs = self.get_images(image_id.tolist(), ann_id, False, image_size)

            imgs = torch.Tensor(imgs)
            imgs = imgs.permute(0, 3, 1, 2)
            imgs /= 255
        else:
            imgs = None
            messages = logs.message

        try:
            class_labels = [self.class_map[idx] for idx in true_class]
        except KeyError:
            class_labels = None

        phase = "train" if is_train else "val"

        self.writer.add_embedding(
            messages,
            metadata=class_labels,
            label_img=imgs,
            global_step=global_step,
            tag=f"{phase}/message",
        )

    def log_messages_distribution(
            self, logs: Interaction, phase: str, global_step: int
    ):
        """
        Logs the messages as an embedding
        Args:
            logs:
            phase:
            global_step:

        Returns:

        """

        var = torch.std(logs.message.float(), dim=1)
        mean = torch.mean(logs.message.float(), dim=1)

        self.writer.add_histogram(
            tag=f"{phase}/message/std",
            values=var,
            global_step=global_step,
        )

        self.writer.add_histogram(
            tag=f"{phase}/message/mean",
            values=mean,
            global_step=global_step,
        )

        self.writer.add_histogram(
            tag=f"{phase}/message/dist",
            values=logs.message,
            global_step=global_step,
        )

    def log(self, loss: float, logs: Interaction, is_training: bool):

        if is_training:
            global_step = self.train_gs
            phase = "train"
            self.train_gs += 1
        else:
            global_step = self.val_gs
            phase = "val"
            self.val_gs += 1

        self.save_gs()
        self.log_metrics(logs, phase, global_step, loss)
        self.log_receiver_output(logs.receiver_output, phase, global_step)
        self.log_labels(logs, phase, global_step)
        self.log_messages_distribution(logs, phase, global_step)


class InteractionCSV(Callback):
    def __init__(
            self,
            tensorboard_dir: str,
            loggers: Dict[str, LoggingStrategy] = None,
            val_coco: COCO = None,
    ):
        """
        Callback to log metrics to tensorboard

        """
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.message_file = Path(join(tensorboard_dir, "interactions.csv"))
        self.loggers = loggers
        self.val_coco = val_coco

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
                f.write("/n")

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
                line += f"{';'.join(distractors[idx])}"
                line += f"{';'.join(other_ans[idx])}\n"
                file.write(line)




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
