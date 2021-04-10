from pathlib import Path

import torch

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


def main(params=None):
    opts = parse_arguments(params=params)
    define_project_dir(opts)
    dump_params(opts)

    _, test_data = get_data(opts)

    game = get_game(opts)

    optimizer = core.build_optimizer(game.parameters())

    interaction_saver = InteractionCSV(opts.tensorboard_dir, test_data.dataset.coco)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=test_data,
        validation_data=None,
    )

    if opts.checkpoint_dir is not None:
        trainer.load_from_latest(Path(opts.checkpoint_dir))

    loss, logs = trainer.train_eval()
    interaction_saver.on_test_end(loss, logs, 0)
    console.log("Test is over")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
