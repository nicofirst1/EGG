from pathlib import Path

import torch

from agents.prolonet_agent import DeepProLoNet
from egg.core.early_stopping import EarlyStopper, EarlyStopperAccuracy

from egg import core
from egg.core import (
    LoggingStrategy,
    ProgressBarLogger, ConsoleLogger, CheckpointSaver,
)
from egg.zoo.coco_game.archs.heads import initialize_model, get_vision_dim
from egg.zoo.coco_game.archs.receiver import build_receiver
from egg.zoo.coco_game.archs.sender import build_sender
from egg.zoo.coco_game.custom_callbacks import CustomEarlyStopperAccuracy, InteractionCSV, SyncLogging, \
    TensorboardLogger
from egg.zoo.coco_game.dataset import get_data
from egg.zoo.coco_game.losses import final_loss
from egg.zoo.coco_game.utils.dataset_utils import get_dummy_data, split_dataset
from egg.zoo.coco_game.utils.parsers import parse_arguments
from egg.zoo.coco_game.utils.utils import (
    console,
    define_project_dir,
    dump_params, get_images,
)


def get_game(opts, is_test=False):
    ######################################
    #   Sender receiver modules
    ######################################
    model = initialize_model()

    receiver = build_receiver(feature_extractor=model, opts=opts)
    sender = build_sender(feature_extractor=model, opts=opts)

    ######################################
    #   Sender receiver wrappers
    ######################################
    # sender = core.RnnSenderReinforce(
    #     sender,
    #     vocab_size=opts.vocab_size,
    #     embed_dim=opts.sender_embedding,
    #     hidden_size=opts.sender_hidden,
    #     max_len=opts.max_len,
    #     num_layers=opts.sender_num_layers,
    #     cell=opts.sender_cell_type,
    # )
    sender= DeepProLoNet(input_dim=get_vision_dim(),
                         output_dim=2,
                         use_gpu=not opts.no_cuda,
                         vocab_size=opts.vocab_size,
                         max_len=opts.max_len,
                         image_processor=sender,
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
    if opts.use_tensorboard_logger or is_test:
        train_log = SyncLogging(logging_step=opts.train_logging_step)
        val_log = SyncLogging(logging_step=opts.val_logging_step)
    else:
        train_log = LoggingStrategy().minimal()
        val_log = LoggingStrategy().minimal()

    loggers=dict(train=train_log, val=val_log)

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss=final_loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0,
        train_logging_strategy=train_log,
        test_logging_strategy=val_log,
    )
    return game, loggers


def main(params=None):
    opts = parse_arguments(params=params)
    define_project_dir(opts)
    dump_params(opts)

    train_data, val_data = get_data(opts)

    game, loggers = get_game(opts)

    optimizer = core.build_optimizer(game.parameters())
    rl_optimizer = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=opts.decay_rate
    )

    callbacks = [
        CustomEarlyStopperAccuracy(min_threshold=0.6, min_increase=0.01, field_name="accuracy"),
        ConsoleLogger(print_train_loss=True, as_json=True),
        CheckpointSaver(checkpoint_path=opts.checkpoint_dir, max_checkpoints=3, prefix="checkpoint"),
    ]

    if opts.use_progress_bar:
        callbacks += [
            ProgressBarLogger(
                n_epochs=opts.n_epochs,
                train_data_len=len(train_data)*2,
                val_data_len=len(val_data),
                use_info_table=False,
            ),
        ]

    if opts.use_tensorboard_logger:
        get_imgs = get_images(train_data.dataset.get_images, val_data.dataset.get_images)

        callbacks+=[
            TensorboardLogger(
                tensorboard_dir=opts.tensorboard_dir,
                resume_training=opts.resume_training,
                loggers=loggers,
                game=game,
                class_map={
                    k: v["name"] for k, v in train_data.dataset.coco.cats.items()
                },
                get_image_method=get_imgs,
                hparams=vars(opts),
            )
        ]

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=val_data,
        callbacks=callbacks,
        optimizer_scheduler=rl_optimizer,
    )

    if opts.checkpoint_dir is not None:
        trainer.load_from_latest(Path(opts.checkpoint_dir))

    trainer.train(n_epochs=opts.n_epochs)
    console.log("Train is over")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
