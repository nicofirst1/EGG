import argparse
import uuid
from os.path import join
from pathlib import Path

import torch

from egg import core
from egg.core import CheckpointSaver, ProgressBarLogger
from egg.zoo.coco_game.archs.heads import initialize_model
from egg.zoo.coco_game.archs.receiver import build_receiver
from egg.zoo.coco_game.archs.sender import VisionSender
from egg.zoo.coco_game.custom_logging import TensorboardLogger, RandomLogging
from egg.zoo.coco_game.dataset import get_data
from egg.zoo.coco_game.losses import loss_init
from egg.zoo.coco_game.utils import console, dump_params, get_images


def parse_arguments(params=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/home/dizzi/Desktop/coco",
        help="Data root folder to coco",
    )

    parser.add_argument(
        "--use_rich_traceback",
        default=False,
        action="store_true",
        help="If to use the traceback provided by the rich library",
    )

    parser.add_argument(
        "--resume_training",
        default=False,
        action="store_true",
        help="Resume training loading models from '--checkpoint_dir'",
    )

    #################################################
    # Loss
    #################################################

    parser.add_argument(
        "--cross_lambda",
        type=float,
        default=1,
        help="Weight for cross entropy loss for classification task.",
    )

    #################################################
    # Vision model
    #################################################
    parser.add_argument(
        "--head_choice",
        type=str,
        default="single",
        help="Choose the receiver box head module: single",
        choices=["single"],
    )
    parser.add_argument(
        "--image_type",
        type=str,
        default="seg",
        help="Choose the sender input type: seg(mented), image, both",
        choices=["seg", "img", "both"],
    )

    parser.add_argument(
        "--image_union",
        type=str,
        default="mul",
        help="When 'image_type==both', how to aggregate infos from both images",
        choices=["cat", "mul"],
    )
    parser.add_argument(
        "--image_resize",
        type=int,
        default=224,
        help="Size of pretrain input image. Minimum is 224",
    )

    #################################################
    # LOG
    #################################################

    parser.add_argument(
        "--log_dir",
        default="./Logs",
        help="Log dir to save all logs",
    )

    parser.add_argument(
        "--log_dir_uid",
        default=f"{str(uuid.uuid4())[:8]}",
        help="Log subdir name where to save single runs",
    )

    parser.add_argument(
        "--train_log_prob",
        type=float,
        default=0.01,
        help="Percentage of training interaction to save",
    )

    parser.add_argument(
        "--test_log_prob",
        type=float,
        default=0.03,
        help="Percentage of test interaction to save",
    )

    parser.add_argument(
        "--train_logging_step",
        type=int,
        default=50,
        help="Number of steps (in batches) before logging during training ",
    )

    parser.add_argument(
        "--test_logging_step",
        type=int,
        default=20,
        help="Number of steps (in batches) before logging during testing",
    )

    #################################################
    # dataLoader args
    #################################################

    parser.add_argument(
        "--min_area",
        type=float,
        default=0,
        help="Minimum percentage of the total image area for object. ",
    )

    parser.add_argument(
        "--skip_first",
        type=int,
        default=5,
        help="Number of first classes to skip. Default 5 bc the first 5 classes are over represented in coco",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=75,
        help="Number of classes to use, 80 for all",
    )

    parser.add_argument(
        "--train_data_perc",
        type=float,
        default=1,
        help="Size of the coco train dataset to be used",
    )
    parser.add_argument(
        "--test_data_perc",
        type=float,
        default=1,
        help="Size of the coco test dataset to be used",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in the dataloader",
    )

    #################################################
    # rnn params
    #################################################

    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=16,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=16,
        help="Size of the hidden layer of Receiver (default: 10)",
    )

    parser.add_argument(
        "--receiver_num_layers",
        type=int,
        default=2,
        help="Number of rnn layers for receiver",
    )

    parser.add_argument(
        "--receiver_cell_type",
        type=str,
        default="lstm",
        choices=["rnn", "lstm", "gru"],
        help="Type of RNN cell for receiver",
    )

    parser.add_argument(
        "--sender_num_layers",
        type=int,
        default=2,
        help="Number of rnn layers for sender",
    )

    parser.add_argument(
        "--sender_cell_type",
        type=str,
        default="lstm",
        choices=["rnn", "lstm", "gru"],
        help="Type of RNN cell for sender",
    )

    parser.add_argument(
        "--box_head_hidden",
        type=int,
        default=32,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=16,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=16,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )

    #################################################
    # parser manipulation
    #################################################

    # add core opt and print
    opt = core.init(parser, params=params)
    console.log(sorted(vars(opt).items()))

    if opt.use_rich_traceback:
        from rich.traceback import install

        install()

    # assert the number of classes is less than 90-skip_first
    assert (
            opt.num_classes + opt.skip_first <= 80
    ), f"The number of classes plus the skip must be less than 90, currently {opt.num_classes + opt.skip_first} "

    assert opt.image_resize>=224, "The size of the image must be minimum 224"
    return opt


def get_game(feat_extractor, opts):
    ######################################
    #   Sender receiver modules
    ######################################
    sender = VisionSender(
        feat_extractor,
        image_size=opts.image_resize,
        image_type=opts.image_type,
        image_union=opts.image_union,
        n_hidden=opts.sender_hidden,
    )

    receiver = build_receiver(feature_extractor=feat_extractor, opts=opts)

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
    train_log = RandomLogging(
        logging_step=opts.train_logging_step, store_prob=opts.train_log_prob
    )
    test_log = RandomLogging(
        logging_step=opts.test_logging_step, store_prob=opts.test_log_prob
    )

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss=loss_init(
            cross_lambda=opts.cross_lambda,
            batch_size=opts.batch_size,
        ),
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0,
        train_logging_strategy=train_log,
        test_logging_strategy=test_log,
    )
    return game, dict(train=train_log, test=test_log)


def define_project_dir(opts):
    """
    Define the dir tree as:
    -log_dir

    --log_dir_uid-1
    ---checkpoint_dir
    ---tensorboard_dir
    ---interactions_path

    --log_dir_uid-2
    ---checkpoint_dir
    ---tensorboard_dir
    ---interactions_path

    ...

    """

    console.log(f"New experiment with uuid: '{opts.log_dir_uid}' created ")

    opts.log_dir_uid = join(opts.log_dir, opts.log_dir_uid)
    # make log dir root for logging paths
    opts.checkpoint_dir = join(opts.log_dir_uid, opts.checkpoint_dir)
    opts.tensorboard_dir = join(opts.log_dir_uid, opts.tensorboard_dir)


def main(params=None):
    opts = parse_arguments(params=params)
    define_project_dir(opts)
    dump_params(opts)
    model = initialize_model()

    game, loggers = get_game(model, opts)

    train, test = get_data( opts)

    optimizer = core.build_optimizer(game.parameters())


    get_imgs = get_images(train.dataset.get_images, test.dataset.get_images)

    callbacks = [
        ProgressBarLogger(
            n_epochs=opts.n_epochs,
            train_data_len=len(train),
            test_data_len=len(test),
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
            game=game,
            class_map={k: v["name"] for k, v in train.dataset.coco.cats.items()},
            get_image_method=get_imgs
        ),
    ]

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train,
        validation_data=test,
        callbacks=callbacks,
    )
    if opts.resume_training:
        trainer.load_from_latest(Path(opts.checkpoint_dir))

    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
