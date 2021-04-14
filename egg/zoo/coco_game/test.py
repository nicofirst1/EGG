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

    tmp=vars(opts)

    for k,v in update_dict.items():
        tmp[k]=v

    opts=Namespace(**tmp)
    return opts


def main(params=None):
    opts = parse_arguments(params=params)


    params = load_params("/home/dizzi/Documents/Egg_expermients/median_runs/Best_Both_Median/0f33c569/params.json")
    opts=update_namespace(opts, params)
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

    trainer.load_from_latest(
        Path("/home/dizzi/Documents/Egg_expermients/median_runs/Best_Both_Median/0f33c569/best_both_median_seed_nest_out"))

    for _ in track(range(1), "Testing..."):
        loss, logs = trainer.train_eval()
        interaction_saver.on_test_end(loss, logs, 0)
    console.log("Test is over")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
