import random
from os.path import join
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from egg.core import Callback, Interaction, LoggingStrategy
from egg.zoo.coco_game.archs.receiver import Receiver
from egg.zoo.coco_game.archs.sender import VisionSender
from egg.zoo.coco_game.utils import get_labels


class TensorboardLogger(Callback):
    def __init__(
            self,
            tensorboard_dir: str,
            loggers: Dict[str, LoggingStrategy],
            train_logging_step: int = 50,
            test_logging_step: int = 20,
            resume_training: bool = False,
            start_epoch: int = 0,
            game: torch.nn.Module = None,
            class_map: Dict[int, str] = {},
            get_image_method=None,
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
            start_epoch: the epoch to start with
        """
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.gs_file = join(tensorboard_dir, "gs.txt")
        self.train_log_step = train_logging_step
        self.test_log_step = test_logging_step
        self.train_gs = 0
        self.test_gs = 0
        self.loggers = loggers
        self.epoch = start_epoch

        self.game = game
        self.class_map = class_map
        self.get_images = get_image_method

        self.embeddings_log_step = 5

        if resume_training:
            try:
                self.get_gs()
            except FileNotFoundError:
                pass

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

        self.log_precision_recall(logs, phase="train", global_step=self.epoch)
        self.log_messages_embedding(logs, is_train=True, global_step=self.epoch)

        self.writer.add_scalar("epoch", self.epoch, global_step=self.train_gs)
        self.loggers["train"].cur_batch = 0
        self.epoch += 1

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.log_precision_recall(logs, phase="test", global_step=self.epoch)
        self.log_messages_embedding(logs, is_train=False, global_step=self.epoch)

        self.loggers["test"].cur_batch = 0

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

        for k, v in metrics.items():
            self.writer.add_scalar(
                tag=f"{phase}/{k}", scalar_value=v.mean(), global_step=global_step
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
        self.game = None

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

        sender: VisionSender = self.game.sender.agent
        receiver: Receiver = self.game.receiver.agent

        # extract modules
        pretrained = sender.vision
        conv_sender = sender.cat_flat.conv
        conv_receiver = receiver.cat_flat.conv

        # port to cpu
        pretrained = pretrained.cpu()
        conv_sender = conv_sender.cpu().eval()
        conv_receiver = conv_receiver.cpu().eval()

        # get info
        batch_size = logs.labels.shape[0]
        idx = random.randint(0, batch_size)

        # get random images
        sender_img = logs.sender_input[idx]
        receiver_img = logs.receiver_input[idx]

        # split img/seg
        img_size = sender_img.shape[1]
        sender_seg = sender_img[:, :, img_size:]
        sender_img = sender_img[:, :, :img_size]

        # add batch size dimension
        sender_seg = sender_seg.unsqueeze(dim=0)
        sender_img = sender_img.unsqueeze(dim=0)
        receiver_img = receiver_img.unsqueeze(dim=0)

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        pretrained[0].register_forward_hook(get_activation("conv1"))
        conv_sender[0].register_forward_hook(get_activation("sender"))
        conv_sender[0].register_forward_hook(get_activation("receiver"))
        # run trough pretrained
        sender_seg = pretrained(sender_seg)
        sender_img = pretrained(sender_img)
        receiver_img = pretrained(receiver_img)

        # run trough conv
        sender_seg = conv_sender(sender_seg)
        sender_img = conv_sender(sender_img)
        receiver_img = conv_receiver(receiver_img)

        # remove batch dimension
        sender_seg = sender_seg.squeeze(dim=0)
        sender_img = sender_img.squeeze(dim=0)
        receiver_img = receiver_img.squeeze(dim=0)

        a = 1

        self.writer.add_image(
            tag=f"{phase}/conv/sender",
            img_tensor=sender_seg,
            global_step=global_step,
            dataformats="HWC",
        )
        self.writer.add_image(
            tag=f"{phase}/conv/receiver",
            img_tensor=sender_img,
            global_step=global_step,
            dataformats="HWC",
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
        Args:
            logs:
            phase:
            global_step:

        Returns:

        """

        if global_step % self.embeddings_log_step != 0:
            return

        true_class, image_id = get_labels(logs.labels)

        if self.get_images is not None:
            # sample down the number of images to load to 200
            to_log = random.sample(range(true_class.shape[0]), k=200)
            image_id = image_id[to_log]
            true_class = true_class[to_log]
            messages=logs.message[to_log]

            if is_train:
                imgs = self.get_images(image_id.tolist(), True, (100, 100))
            else:
                imgs = self.get_images(image_id.tolist(), False, (100, 100))

            imgs = torch.Tensor(imgs)
            imgs = imgs.permute(0, 3, 1, 2)
            imgs /= 255
        else:
            imgs = None
            messages=logs.message

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
        # self.log_conv_filter(logs, phase, global_step)

        # if self.game is not None:
        #     self.log_graphs(logs)


def get_single_label(labels, idx):
    bbox, classes = get_labels(labels)
    bbox = bbox[idx]
    classes = int(classes[idx])

    return bbox, classes
