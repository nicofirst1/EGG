from pathlib import Path

import torch

from egg import core
from egg.core import CheckpointSaver, ProgressBarLogger
from egg.zoo.coco_game.archs.heads import initialize_model
from egg.zoo.coco_game.archs.sender import build_sender
from egg.zoo.coco_game.custom_logging import RandomLogging, TensorboardLogger, RlScheduler
from egg.zoo.coco_game.dataset import get_data
from egg.zoo.coco_game.losses import loss_init
from egg.zoo.coco_game.pretrain.game import PretrainGame
from egg.zoo.coco_game.utils.hypertune import hypertune
from egg.zoo.coco_game.utils.utils import console, dump_params, parse_arguments, \
    define_project_dir, get_class_weight


@hypertune
def pretrain(params=None):
    opts = parse_arguments(params=params)
    define_project_dir(opts)
    dump_params(opts)
    model = initialize_model()
    sender = build_sender(feature_extractor=model, opts=opts, pretrain=True)

    train_data, test_data = get_data(opts)

    class_weights = get_class_weight(train_data, opts)

    optimizer = core.build_optimizer(sender.parameters())
    rl_optimizer = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opts.decay_rate)

    criterion = loss_init(
        cross_lambda=opts.cross_lambda,
        kl_lambda=opts.kl_lambda,
        batch_size=opts.batch_size,
        class_weights=class_weights,
    )

    train_log = RandomLogging(
        logging_step=opts.train_logging_step, store_prob=opts.train_log_prob
    )
    test_log = RandomLogging(
        logging_step=opts.test_logging_step, store_prob=opts.test_log_prob
    )
    loggers = dict(train=train_log, test=test_log)

    callbacks = [
        ProgressBarLogger(
            n_epochs=opts.n_epochs,
            train_data_len=len(train_data),
            test_data_len=len(test_data),
            use_info_table=False,
        ),
        CheckpointSaver(
            checkpoint_path=opts.checkpoint_dir,
            checkpoint_freq=opts.checkpoint_freq,
            prefix="epoch",
            max_checkpoints=10,
        ),
        TensorboardLogger(
            tensorboard_dir=opts.tensorboard_dir,
            train_logging_step=opts.train_logging_step,
            test_logging_step=opts.test_logging_step,
            resume_training=opts.resume_training,
            loggers=loggers,
            game=None,
            class_map={},
            get_image_method=None,
            hparams=vars(opts),
        ),
        RlScheduler(rl_optimizer=rl_optimizer),
    ]

    game = PretrainGame(sender, criterion, opts, train_log, test_log)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=test_data,
        callbacks=callbacks,
    )
    if opts.resume_training:
        trainer.load_from_latest(Path(opts.checkpoint_dir))

    trainer.train(n_epochs=opts.n_epochs)
    console.log("Sender Pretrain is over")


if __name__ == "__main__":
    pretrain()
