import copy
import random
from os.path import join
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from egg.core import Callback, Interaction, LoggingStrategy
from egg.zoo.coco_game.utils.utils import get_labels


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
            loggers: Dict[str, LoggingStrategy],
            train_logging_step: int = 50,
            test_logging_step: int = 20,
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
            test_logging_step:  number of batches to log in testing
            resume_training: if the train is resumed, load previous global steps
            game: A torch module used to log the graph
        """
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.gs_file = join(tensorboard_dir, "gs.txt")
        self.train_log_step = train_logging_step
        self.test_log_step = test_logging_step
        self.train_gs = 0
        self.test_gs = 0
        self.loggers = loggers

        self.game = game
        self.class_map = class_map
        self.get_images = get_image_method

        self.embeddings_log_step = 3
        self.log_conv = False
        self.log_graph = False

        self.hparam = self.filter_hparam(hparams)

        if resume_training:
            try:
                self.get_gs()
            except FileNotFoundError:
                pass

    @staticmethod
    def filter_hparam(hparam):
        allowed_types = [int, float, str, bool, torch.Tensor]
        hparam = {k: v for k, v in hparam.items() if any([isinstance(v, t) for t in allowed_types])}
        return hparam

    def get_gs(self):
        """
        Read global steps from file
        """
        with open(self.gs_file, "r+") as f:
            line = f.read()

        self.train_gs = int(line.split(",")[0])
        self.test_gs = int(line.split(",")[1])

    def save_gs(self):
        """
        Dump global steps from file
        """
        with open(self.gs_file, "w+") as f:
            f.write(f"{self.train_gs},{self.test_gs}")

    def on_train_end(self):
        self.writer.close()

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):

        self.log_precision_recall(logs, phase="train", global_step=epoch)
        self.log_messages_embedding(logs, is_train=True, global_step=epoch)
        if self.log_conv:
            self.log_conv_filter(logs, phase="train", global_step=epoch)
        if self.log_graph:
            self.log_graphs(logs)
            self.log_graph = False

        self.writer.add_scalar("epoch", epoch, global_step=self.train_gs)
        self.loggers["train"].cur_batch = 0

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.log_precision_recall(logs, phase="test", global_step=epoch)
        self.log_messages_embedding(logs, is_train=False, global_step=epoch)
        self.log_hparams(logs,loss)
        if self.log_conv:
            self.log_conv_filter(logs, phase="train", global_step=epoch)

        self.loggers["test"].cur_batch = 0
        self.log_conv = False

    def on_batch_end(
            self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):

        if batch_id != 0:
            if batch_id % self.train_log_step == 0 and is_training:
                self.log(loss.detach(), logs, is_training)
            if batch_id % self.test_log_step == 0 and not is_training:
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
        metrics['loss'] = loss

        self.writer.add_hparams(hparam_dict=self.hparam, metric_dict=metrics, run_name="hparams")

    def log_labels(self, logs: Interaction, phase: str, global_step: int):
        """
        Logs statistic about the labels such as the class and the bounding boxes
        """
        labels = logs.labels

        # classes
        true_classes, _ = get_labels(labels)

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
                            pretrained[i][j].register_forward_hook(get_activation(f"conv{i + j}"))

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
        true_classes, _ = get_labels(logs.labels)
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

    def log_messages_embedding(self, logs: Interaction, is_train: bool, global_step: int):
        """
        Logs the messages as an embedding
        """

        if global_step % self.embeddings_log_step != 0:
            return

        true_class, image_id = get_labels(logs.labels)

        if self.get_images is not None:
            # sample down the number of images to load to 200
            to_log = random.sample(range(true_class.shape[0]), k=min(200, true_class.shape[0]))
            image_id = image_id[to_log]
            true_class = true_class[to_log]
            messages = logs.message[to_log]

            if is_train:
                imgs = self.get_images(image_id.tolist(), True, (100, 100))
            else:
                imgs = self.get_images(image_id.tolist(), False, (100, 100))

            imgs = torch.Tensor(imgs)
            imgs = imgs.permute(0, 3, 1, 2)
            imgs /= 255
        else:
            imgs = None
            messages = logs.message

        try:
            class_labels = [self.class_map[idx] for idx in true_class.tolist()]
        except KeyError:
            class_labels = None

        phase = "train" if is_train else "test"

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
            global_step = self.test_gs
            phase = "test"
            self.test_gs += 1

        self.save_gs()
        self.log_metrics(logs, phase, global_step, loss)
        self.log_receiver_output(logs.receiver_output, phase, global_step)
        self.log_labels(logs, phase, global_step)
        self.log_messages_distribution(logs, phase, global_step)


def get_single_label(labels, idx):
    bbox, classes = get_labels(labels)
    bbox = bbox[idx]
    classes = int(classes[idx])

    return bbox, classes
