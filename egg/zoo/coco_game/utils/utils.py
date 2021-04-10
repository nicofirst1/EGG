import argparse
import json
import re
from copy import copy
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict

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
    target_position = labels[:, 0, 0]
    label_class = labels[:, :, 1]
    label_img_id = labels[:, 0, 2]
    ann_id = labels[:, :, 3]
    res = dict(
        target_position=target_position,
        class_id=label_class,
        image_id=label_img_id,
        ann_id=ann_id,
    )
    return res


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


    opts.log_dir_uid = join(opts.log_dir, opts.log_dir_uid)
    # make log dir root for logging paths
    opts.tensorboard_dir = join(opts.log_dir_uid, opts.tensorboard_dir)
    console.log(f"New experiment with uuid: '{opts.log_dir_uid}' created ")

    if opts.checkpoint_dir is not None:
        opts.checkpoint_dir = join(opts.log_dir_uid, opts.checkpoint_dir)


