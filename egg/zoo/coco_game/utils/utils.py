import argparse
import json
import re
from copy import copy
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Tuple

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


def get_labels(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Only function to be used to extract labels information
    """
    label_class = labels[:, 0]
    label_img_id = labels[:, 1]
    return label_class, label_img_id


def get_images(train_method, test_method):
    def inner(image_ids, is_training, img_size):
        if is_training:
            return train_method(image_ids, img_size)
        else:
            return test_method(image_ids, img_size)

    return inner



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
