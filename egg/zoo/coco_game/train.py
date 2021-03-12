from pathlib import Path

import torch

from egg import core
from egg.core import (
    ConsoleLogger,
    LoggingStrategy,
    ProgressBarLogger,
    RnnSenderReinforce,
    SenderReceiverRnnReinforce,
)
from egg.zoo.coco_game.archs.heads import initialize_model
from egg.zoo.coco_game.archs.receiver import build_receiver
from egg.zoo.coco_game.archs.sender import build_sender
from egg.zoo.coco_game.custom_callbacks import EarlyStopperAccuracy, RlScheduler
from egg.zoo.coco_game.dataset import get_data
from egg.zoo.coco_game.losses import final_loss
from egg.zoo.coco_game.utils.dataset_utils import get_dummy_data
from egg.zoo.coco_game.utils.parsers import parse_arguments
from egg.zoo.coco_game.utils.utils import (
    console,
    define_project_dir,
    dump_params,
    get_class_weight,
)


def get_game(feat_extractor, opts):
    ######################################
    #   Sender receiver modules
    ######################################

    receiver = build_receiver(feature_extractor=feat_extractor, opts=opts)
    sender = build_sender(feature_extractor=feat_extractor, opts=opts)

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
    model = initialize_model()

    train_data, val_data = get_data(opts)

    if opts.use_dummy_val:
        console.log("Using Dummy dataset as validation")
        dummy = get_dummy_data(len(val_data.dataset), opts)
        val_data = dummy
    elif opts.use_train_val:
        console.log("Using train dataset as validation")
        val_data = train_data

    game = get_game(model, opts)

    optimizer = core.build_optimizer(game.parameters())
    rl_optimizer = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=opts.decay_rate
    )

    callbacks = [
        #todo: use egg rl scheduler
        RlScheduler(rl_optimizer=rl_optimizer),
        EarlyStopperAccuracy(max_threshold=0.6, min_increase=0.01),
        ConsoleLogger(print_train_loss=True, as_json=True),
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
    )
    # todo: try to remove resume_training
    if opts.resume_training:
        trainer.load_from_latest(Path(opts.checkpoint_dir))

    trainer.train(n_epochs=opts.n_epochs)
    console.log("Train is over")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
