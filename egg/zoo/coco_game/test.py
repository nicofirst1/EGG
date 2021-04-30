import sys
from argparse import Namespace
from pathlib import Path

import torch
from rich.progress import track

from egg import core
from egg.zoo.coco_game.custom_callbacks import InteractionCSV
from egg.zoo.coco_game.dataset import get_data
from egg.zoo.coco_game.train import get_game
from egg.zoo.coco_game.utils.parsers import parse_arguments
from egg.zoo.coco_game.utils.utils import (
    console,
    define_project_dir,
    dump_params,
)


def load_params(params_path):
    # parse x:
    with open(params_path, "r") as file:
        lines = file.readline()

    lines = lines.replace("null", "None")
    lines = lines.replace("false", "False")
    lines = lines.replace("true", "True")
    lines = eval(lines)
    lines.pop("log_dir")
    lines.pop("log_dir_uid")
    lines.pop("data_root")
    lines.pop("checkpoint_dir")
    lines.pop("tensorboard_dir")

    return lines


def update_namespace(opts: Namespace, update_dict):
    tmp = vars(opts)

    for k, v in update_dict.items():
        tmp[k] = v

    opts = Namespace(**tmp)
    return opts


def main(is_seg):
    params = sys.argv[1:]+['--random_seed', '33', '--data_seed', '42']

    opts = parse_arguments(params=params)

    both_path = "0f33c569"
    seg_path = "b407ecb5"

    if is_seg:

        file_path = f"/home/dizzi/Documents/Egg_expermients/median_runs/{seg_path}/"
    else:
        file_path = f"/home/dizzi/Documents/Egg_expermients/median_runs/{both_path}/"

    epochs = 1

    params = load_params(f"{file_path}params.json")
    opts = update_namespace(opts, params)

    opts.random_seed = 33
    opts.data_seed = 42

    if is_seg:
        opts.log_dir_uid = "seg"
    else:
        opts.log_dir_uid = "both"

    console.log(sorted(vars(opts).items()))

    define_project_dir(opts)

    dump_params(opts)

    _, test_data = get_data(opts)

    game = get_game(opts, is_test=True)

    optimizer = core.build_optimizer(game.parameters())

    interaction_saver = InteractionCSV(opts.tensorboard_dir, test_data.dataset.coco)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=test_data,
        validation_data=None,
    )

    trainer.load_from_latest(Path(file_path))

    for _ in track(range(epochs), "Testing..."):
        loss, logs = trainer.train_eval()
        interaction_saver.on_test_end(loss, logs, 0)
    console.log("Test is over")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    main(is_seg=False)
    main(is_seg=True)
