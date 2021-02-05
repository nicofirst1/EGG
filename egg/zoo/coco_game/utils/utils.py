import argparse
import json
import os
import re
import uuid
from copy import copy
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from rich.console import Console

from egg import core
from egg.core.callbacks import Checkpoint
from egg.zoo.coco_game.archs import FLAT_CHOICES, HEAD_CHOICES

console = Console()


def load_last_chk(checkpoint_dir: str) -> Checkpoint:
    def alphanum_key(s):
        """Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """

        def tryint(s):
            try:
                return int(s)
            except:
                return s

        return [tryint(c) for c in re.split("([0-9]+)", s)]

    files = [f for f in listdir(checkpoint_dir) if isfile(join(checkpoint_dir, f))]

    files.sort(key=alphanum_key)

    f_path = join(checkpoint_dir, files[-1])

    chk = torch.load(f_path)

    return chk


def dump_params(opts):
    """
    Dumps the opts into the logdir
    """
    file_path = join(opts.log_dir_uid, "params.json")
    Path(opts.log_dir_uid).mkdir(parents=True, exist_ok=True)

    to_dump = copy(vars(opts))
    to_dump.pop("device")
    to_dump.pop("distributed_context")

    with open(file_path, "w") as fp:
        json.dump(to_dump, fp)


def get_labels(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Only function to be used to extract labels information
    """
    label_class = labels[:, 0]
    label_img_id = labels[:, 1]
    ann_id = labels[:, 2]
    res = dict(
        class_id=label_class,
        image_id=label_img_id,
        ann_id=ann_id,
    )
    return res


def get_images(train_method, val_method):
    def inner(
        image_ids: List[int],
        image_ann_ids: List[int],
        is_training: bool,
        img_size: Tuple[int, int],
    ):
        if is_training:
            return train_method(image_ids, image_ann_ids, img_size)
        else:
            return val_method(image_ids, image_ann_ids, img_size)

    return inner


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def define_project_dir(opts):
    """
    Define the dir tree as:
    -log_dir

    --log_dir_uid-1
    ---checkpoint_dir
    ---tensorboard_dir
    ---interactions_path

    --log_dir_uid-2
    ---checkpoint_dir
    ---tensorboard_dir
    ---interactions_path

    ...

    """

    console.log(f"New experiment with uuid: '{opts.log_dir_uid}' created ")

    opts.log_dir_uid = join(opts.log_dir, opts.log_dir_uid)
    # make log dir root for logging paths
    if opts.checkpoint_dir is not None:
        opts.checkpoint_dir = join(opts.log_dir_uid, opts.checkpoint_dir)
    opts.tensorboard_dir = join(opts.log_dir_uid, opts.tensorboard_dir)


def get_class_weight(train, opts):
    if opts.use_class_weights:
        class_weights = train.dataset.get_class_weights()
        # transform from dict to sorted tensor
        class_weights = [x[1] for x in sorted(class_weights.items())]
        class_weights = torch.Tensor(class_weights)
        class_weights = class_weights.to(opts.device)

    else:
        class_weights = None

    return class_weights


def parse_arguments(params=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        help="Data root folder to coco",
    )

    parser.add_argument(
        "--use_rich_traceback",
        default=False,
        action="store_true",
        help="If to use the traceback provided by the rich library",
    )

    parser.add_argument(
        "--resume_training",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Resume training loading models from '--checkpoint_dir'",
    )

    parser.add_argument(
        "--sender_pretrain",
        type=str,
        nargs="?",
        const=True,
        default="",
        help="Path for pretrained sender weights",
    )

    #################################################
    # Loss
    #################################################

    parser.add_argument(
        "--lambda_cross",
        type=float,
        default=1,
        help="Weight for cross entropy loss for classification task.",
    )

    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=0,
        help="Weight for Kullback-Leibler divergence loss for classification task.",
    )

    parser.add_argument(
        "--lambda_f",
        type=float,
        default=0,
        help="Weight for Focal loss for classification task.",
    )

    parser.add_argument(
        "--use_class_weights",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Extract class weights from the dataset and pass them to the loss function",
    )

    #################################################
    # Vision model
    #################################################
    parser.add_argument(
        "--head_choice",
        type=str,
        default="simple",
        help="Choose the receiver box head module",
        choices=list(HEAD_CHOICES.keys()),
    )

    parser.add_argument(
        "--flat_choice_sender",
        type=str,
        default="AvgPool",
        help="Choose the flat module type",
        choices=list(FLAT_CHOICES.keys()),
    )

    parser.add_argument(
        "--flat_choice_receiver",
        type=str,
        default="AvgPool",
        help="Choose the flat module type",
        choices=list(FLAT_CHOICES.keys()),
    )

    parser.add_argument(
        "--image_type",
        type=str,
        default="both",
        help="Choose the sender input type: seg(mented), image, both",
        choices=["seg", "img", "both"],
    )

    parser.add_argument(
        "--image_union",
        type=str,
        default="mul",
        help="When 'image_type==both', how to aggregate infos from both images",
        choices=["cat", "mul"],
    )
    parser.add_argument(
        "--image_resize",
        type=int,
        default=224,
        help="Size of pretrain input image. Minimum is 224",
    )

    #################################################
    # LOG
    #################################################

    parser.add_argument(
        "--log_dir",
        default="./Logs",
        help="Log dir to save all logs",
    )

    parser.add_argument(
        "--log_dir_uid",
        default=f"{str(uuid.uuid4())[:8]}",
        help="Log subdir name where to save single runs",
    )

    parser.add_argument(
        "--train_log_prob",
        type=float,
        default=0.0,
        help="Percentage of training interaction to save",
    )

    parser.add_argument(
        "--val_log_prob",
        type=float,
        default=0.0,
        help="Percentage of val interaction to save",
    )

    parser.add_argument(
        "--train_logging_step",
        type=int,
        default=300,
        help="Number of steps (in batches) before logging during training ",
    )

    parser.add_argument(
        "--val_logging_step",
        type=int,
        default=30,
        help="Number of steps (in batches) before logging during validaiton",
    )

    #################################################
    # dataLoader args
    #################################################

    parser.add_argument(
        "--min_area",
        type=float,
        default=0,
        help="Minimum percentage of the total image area for object. ",
    )

    parser.add_argument(
        "--skip_first",
        type=int,
        default=0,
        help="Number of first classes to skip. Default 5 bc the first 5 classes are over represented in coco",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=80,
        help="Number of classes to use, 80 for all",
    )

    parser.add_argument(
        "--train_data_perc",
        type=float,
        default=1,
        help="Size of the coco train dataset to be used",
    )
    parser.add_argument(
        "--val_data_perc",
        type=float,
        default=1,
        help="Size of the coco val dataset to be used",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in the dataloader",
    )

    #################################################
    # rnn params
    #################################################

    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.8,
        help="Decay rate for lr ",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=128,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=128,
        help="Size of the hidden layer of Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_receiver_hidden",
        type=int,
        default=None,
        help="Size of the hidden layer of both Sender Receiver (default: None)",
    )

    parser.add_argument(
        "--receiver_num_layers",
        type=int,
        default=1,
        help="Number of rnn layers for receiver",
    )

    parser.add_argument(
        "--receiver_cell_type",
        type=str,
        default="gru",
        choices=["rnn", "lstm", "gru"],
        help="Type of RNN cell for receiver",
    )

    parser.add_argument(
        "--sender_num_layers",
        type=int,
        default=1,
        help="Number of rnn layers for sender",
    )

    parser.add_argument(
        "--sender_cell_type",
        type=str,
        default="gru",
        choices=["rnn", "lstm", "gru"],
        help="Type of RNN cell for sender",
    )

    parser.add_argument(
        "--box_head_hidden",
        type=int,
        default=32,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=16,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=16,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )

    #################################################
    # parser manipulation
    #################################################

    # add core opt and print
    opt = core.init(parser, params=params)

    if opt.sender_receiver_hidden is not None:
        opt.sender_hidden = opt.sender_receiver_hidden
        opt.receiver_hidden = opt.sender_receiver_hidden

    console.log(sorted(vars(opt).items()))

    if opt.use_rich_traceback:
        from rich.traceback import install

        install()

    # assert the number of classes is less than 90-skip_first
    assert (
        opt.num_classes + opt.skip_first <= 80
    ), f"The number of classes plus the skip must be less than 90, currently {opt.num_classes + opt.skip_first} "

    assert opt.image_resize >= 224, "The size of the image must be minimum 224"
    return opt


def load_pretrained_sender(path, sender: torch.nn.Module):
    latest_file, latest_time = None, None

    for file in path.glob("*.tar"):
        creation_time = os.stat(file).st_ctime
        if latest_time is None or creation_time > latest_time:
            latest_file, latest_time = file, creation_time

    if latest_file is not None:
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        console.log(f"# loading trainer state from {latest_file}")
        checkpoint = torch.load(latest_file)
        sender.load_state_dict(checkpoint.model_state_dict)
    else:
        console.log(f"Could not state from {path}")
