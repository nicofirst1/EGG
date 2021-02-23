from pathlib import Path

import torch

from egg import core
from egg.core import CheckpointSaver, ProgressBarLogger
from egg.zoo.coco_game.archs.heads import initialize_model
from egg.zoo.coco_game.archs.sender import build_sender
from egg.zoo.coco_game.custom_logging import (
    SyncLogging,
    RlScheduler,
    TensorboardLogger,
)
from egg.zoo.coco_game.dataset import get_data
from egg.zoo.coco_game.losses import loss_init
from egg.zoo.coco_game.pretrain.game import PretrainGame, SenderSaver
from egg.zoo.coco_game.utils.hypertune import hypertune
from egg.zoo.coco_game.utils.utils import (
    console,
    define_project_dir,
    dump_params,
    get_class_weight,
    parse_arguments,
)


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
    rl_optimizer = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=opts.decay_rate
    )

    criterion = loss_init(
        lambda_cross=opts.lambda_cross,
        lambda_kl=opts.lambda_kl,
        lambda_f=opts.lambda_f,
        batch_size=opts.batch_size,
        class_weights=class_weights,
    )

    train_log = SyncLogging(
        logging_step=opts.train_logging_step, store_prob=opts.train_log_prob
    )
    test_log = SyncLogging(
        logging_step=opts.test_logging_step, store_prob=opts.test_log_prob
    )
    loggers = dict(train=train_log, test=test_log)

    callbacks = [
        ProgressBarLogger(
            n_epochs=opts.n_epochs,
            train_data_len=len(train_data),
            val_data_len=len(test_data),
            use_info_table=False,
        ),
        CheckpointSaver(
            checkpoint_path=opts.checkpoint_dir,
            checkpoint_freq=opts.checkpoint_freq,
            prefix="epoch",
            max_checkpoints=10,
        ),
        SenderSaver(
            sender=sender,
            checkpoint_path=opts.checkpoint_dir + "/Sender",
            checkpoint_freq=opts.checkpoint_freq,
            prefix="epoch",
            max_checkpoints=10,
        ),
        TensorboardLogger(
            tensorboard_dir=opts.tensorboard_dir,
            train_logging_step=opts.train_logging_step,
            val_logging_step=opts.test_logging_step,
            resume_training=opts.resume_training,
            loggers=loggers,
            game=None,
            class_map={k: v["name"] for k, v in train_data.dataset.coco.cats.items()},
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
