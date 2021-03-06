from pathlib import Path

import torch

from egg import core
from egg.core import ConsoleLogger, ProgressBarLogger, LoggingStrategy
from egg.core.baselines import MeanBaseline
from egg.zoo.coco_game.archs.heads import initialize_model
from egg.zoo.coco_game.archs.receiver import build_receiver
from egg.zoo.coco_game.archs.sender import build_sender
from egg.zoo.coco_game.custom_logging import (
    RlScheduler,
    TensorboardLogger,
)
from egg.zoo.coco_game.dataset import get_data
from egg.zoo.coco_game.losses import loss_init
from egg.zoo.coco_game.pretrain.sender_reinforce import (
    CustomSenderReceiverRnnReinforce,
    CustomSenderReinforce,
)
from egg.zoo.coco_game.utils.dataset_utils import get_dummy_data
from egg.zoo.coco_game.utils.hypertune import hypertune
from egg.zoo.coco_game.utils.parsers import parse_arguments
from egg.zoo.coco_game.utils.utils import (
    console,
    define_project_dir,
    dump_params,
    get_class_weight,
    get_images,
)


def get_game(feat_extractor, opts, class_weights=None):
    ######################################
    #   Sender receiver modules
    ######################################

    receiver = build_receiver(feature_extractor=feat_extractor, opts=opts)
    sender = build_sender(feature_extractor=feat_extractor, opts=opts)

    ######################################
    #   Sender receiver wrappers
    ######################################
    sender = CustomSenderReinforce(
        sender,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_embedding,
        hidden_size=opts.sender_hidden,
        max_len=opts.max_len,
        num_layers=opts.sender_num_layers,
        cell=opts.sender_cell_type,
    )
    receiver = core.RnnReceiverDeterministic(
        receiver,
        vocab_size=opts.vocab_size,
        embed_dim=opts.receiver_embedding,
        hidden_size=opts.receiver_hidden,
        num_layers=opts.receiver_num_layers,
        cell=opts.receiver_cell_type,
    )

    ######################################
    #   Game wrapper
    ######################################
    # train_log = SyncLogging(logging_step=opts.train_logging_step)
    # val_log = SyncLogging(logging_step=opts.val_logging_step)

    train_log = LoggingStrategy().minimal()
    val_log = LoggingStrategy().minimal()

    game = CustomSenderReceiverRnnReinforce(
        sender,
        receiver,
        loss=loss_init(
            lambda_cross=opts.lambda_cross,
            lambda_kl=opts.lambda_kl,
            lambda_f=opts.lambda_f,
            batch_size=opts.batch_size,
            class_weights=class_weights,
        ),
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0,
        train_logging_strategy=train_log,
        val_logging_strategy=val_log,
        baseline_type=MeanBaseline,
    )
    return game, dict(train=train_log, val=val_log)


@hypertune
def main(params=None):
    opts = parse_arguments(params=params)
    define_project_dir(opts)
    dump_params(opts)
    model = initialize_model()

    train_data, val_data = get_data(opts)
    dummy = get_dummy_data(len(val_data), 10, opts)
    val_data = dummy

    class_weights = get_class_weight(train_data, opts)
    game, loggers = get_game(model, opts, class_weights=class_weights)

    optimizer = core.build_optimizer(game.parameters())
    rl_optimizer = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=opts.decay_rate
    )

    get_imgs = get_images(train_data.dataset.get_images, val_data.dataset.get_images)

    callbacks = [
        # InteractionCSV(
        #     tensorboard_dir=opts.tensorboard_dir,
        #     loggers=loggers,
        #     val_coco=val_data.dataset.coco,
        # ),
        # CheckpointSaver(
        #     checkpoint_path=opts.checkpoint_dir,
        #     checkpoint_freq=opts.checkpoint_freq,
        #     prefix="epoch",
        #     max_checkpoints=10,
        # ),
        RlScheduler(rl_optimizer=rl_optimizer),
        # EarlyStopperAccuracy(max_threshold=0.6, min_increase=0.01),
        ConsoleLogger(print_train_loss=True, as_json=True),
    ]

    if opts.use_custom_logging:
        clbs = [
            TensorboardLogger(
                tensorboard_dir=opts.tensorboard_dir,
                train_logging_step=opts.train_logging_step,
                val_logging_step=opts.val_logging_step,
                resume_training=opts.resume_training,
                loggers=loggers,
                game=game,
                class_map={
                    k: v["name"] for k, v in train_data.dataset.coco.cats.items()
                },
                get_image_method=get_imgs,
                hparams=vars(opts),
            ), ]

        callbacks += clbs

    if opts.use_progress_bar:
        clbs = [
            ProgressBarLogger(
                n_epochs=opts.n_epochs,
                train_data_len=len(train_data),
                val_data_len=len(val_data),
                use_info_table=False,
            ),

        ]
        callbacks += clbs

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=val_data,
        callbacks=callbacks,
    )
    if opts.resume_training:
        trainer.load_from_latest(Path(opts.checkpoint_dir))

    trainer.train(n_epochs=opts.n_epochs)
    console.log("Train is over")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
