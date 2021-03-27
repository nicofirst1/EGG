from pathlib import Path

import torch

from egg import core
from egg.core import (
    LoggingStrategy,
    ProgressBarLogger,
)
from egg.zoo.coco_game.archs.heads import initialize_model
from egg.zoo.coco_game.archs.receiver import build_receiver
from egg.zoo.coco_game.archs.sender import build_sender
from egg.zoo.coco_game.custom_callbacks import CustomConsoleLogger
from egg.zoo.coco_game.dataset import get_data
from egg.zoo.coco_game.losses import final_loss
from egg.zoo.coco_game.utils.dataset_utils import get_dummy_data, split_dataset
from egg.zoo.coco_game.utils.parsers import parse_arguments
from egg.zoo.coco_game.utils.utils import (
    console,
    define_project_dir,
    dump_params,
)


def get_game(opts):
    ######################################
    #   Sender receiver modules
    ######################################
    model = initialize_model()

    receiver = build_receiver(feature_extractor=model, opts=opts)
    sender = build_sender(feature_extractor=model, opts=opts)

    ######################################
    #   Sender receiver wrappers
    ######################################
    sender = core.RnnSenderReinforce(
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

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss=final_loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0,
        train_logging_strategy=train_log,
        test_logging_strategy=val_log,
    )
    return game


def main(params=None):
    opts = parse_arguments(params=params)
    define_project_dir(opts)
    dump_params(opts)

    train_data, val_data = get_data(opts)

    if opts.use_dummy_val:
        console.log("Using Dummy dataset as validation")
        dummy = get_dummy_data(len(val_data.dataset), opts)
        val_data = dummy
    elif opts.use_train_val:
        console.log("Using train dataset as validation")
        val_data = train_data
    elif opts.use_train_split:
        console.log("Use train split as validation")
        train_data, val_data = split_dataset(train_data)
    elif opts.use_invert_data:
        console.log("Using train as val and val as train")
        d = train_data
        train_data = val_data
        val_data = d

    game = get_game(opts)

    optimizer = core.build_optimizer(game.parameters())
    rl_optimizer = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=opts.decay_rate
    )

    callbacks = [
        # EarlyStopperAccuracy(max_threshold=1.4, min_increase=0.01),
        CustomConsoleLogger(print_train_loss=True, as_json=True),
    ]

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
        optimizer_scheduler=rl_optimizer,
    )

    if opts.checkpoint_dir is not None:
        trainer.load_from_latest(Path(opts.checkpoint_dir))

    trainer.train(n_epochs=opts.n_epochs)
    console.log("Train is over")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
