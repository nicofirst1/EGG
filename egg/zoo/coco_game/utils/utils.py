import argparse
import json
import os
import re
from copy import copy
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from rich.console import Console

from egg.core.callbacks import Checkpoint

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
    true_segment = labels[:, 0, 0]
    label_class = labels[:, :, 1]
    label_img_id = labels[:, 0, 2]
    ann_id = labels[:, :, 3]
    res = dict(
        true_segment=true_segment,
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


def get_true_elems(true_segments, classes, annotations):
    true_classes = []
    true_annotations = []

    for idx in range(len(true_segments)):
        ts = true_segments[idx]
        tc = classes[idx][ts]
        ta = annotations[idx][ts]

        true_classes.append(tc)
        true_annotations.append(ta)

    return true_classes, true_annotations


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
