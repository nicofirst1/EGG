import argparse
import uuid

from egg import core
from egg.zoo.coco_game.archs import FLAT_CHOICES, HEAD_CHOICES
from egg.zoo.coco_game.utils.utils import console, str2bool


def loss_parser(parser):
    parser.add_argument(
        "--lambda_cross",
        type=float,
        default=1,
        help="Weight for cross entropy loss for classification task.",
    )

    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=0,
        help="Weight for Kullback-Leibler divergence loss for classification task.",
    )

    parser.add_argument(
        "--lambda_f",
        type=float,
        default=0,
        help="Weight for Focal loss for classification task.",
    )

    parser.add_argument(
        "--use_class_weights",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Extract class weights from the dataset and pass them to the loss function",
    )


def image_parser(parser):
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


def logs_parser(parser):

    parser.add_argument(
        "--use_rich_traceback",
        default=False,
        action="store_true",
        help="If to use the traceback provided by the rich library",
    )

    parser.add_argument(
        "--use_progress_bar",
        default=False,
        action="store_true",
        help="If to use the traceback provided by the rich library",
    )

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
        default=0.0,
        help="Percentage of training interaction to save",
    )

    parser.add_argument(
        "--val_log_prob",
        type=float,
        default=0.0,
        help="Percentage of val interaction to save",
    )

    parser.add_argument(
        "--train_logging_step",
        type=int,
        default=100,
        help="Number of steps (in batches) before logging during training ",
    )

    parser.add_argument(
        "--val_logging_step",
        type=int,
        default=5,
        help="Number of steps (in batches) before logging during validation",
    )


def dataloader_parsing(parser):
    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Random seed for dataloaders",
    )

    parser.add_argument(
        "--min_area",
        type=float,
        default=0,
        help="Minimum percentage of the total image area for object. ",
    )

    parser.add_argument(
        "--skip_first",
        type=int,
        default=0,
        help="Number of first classes to skip. Default 5 bc the first 5 classes are over represented in coco",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=80,
        help="Number of classes to use, 80 for all",
    )

    parser.add_argument(
        "--train_data_perc",
        type=float,
        default=1,
        help="Size of the coco train dataset to be used",
    )
    parser.add_argument(
        "--val_data_perc",
        type=float,
        default=1,
        help="Size of the coco val dataset to be used",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers in the dataloader",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=1,
        help="Number of distractors per image",
    )
    parser.add_argument(
        "--data_root",
        help="Data root folder to coco",
    )


def egg_arch_parsing(parser):
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=128,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=128,
        help="Size of the hidden layer of Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_receiver_hidden",
        type=int,
        default=None,
        help="Size of the hidden layer of both Sender Receiver (default: None)",
    )

    parser.add_argument(
        "--receiver_num_layers",
        type=int,
        default=1,
        help="Number of rnn layers for receiver",
    )

    parser.add_argument(
        "--receiver_cell_type",
        type=str,
        default="gru",
        choices=["rnn", "lstm", "gru"],
        help="Type of RNN cell for receiver",
    )

    parser.add_argument(
        "--sender_num_layers",
        type=int,
        default=1,
        help="Number of rnn layers for sender",
    )

    parser.add_argument(
        "--sender_cell_type",
        type=str,
        default="gru",
        choices=["rnn", "lstm", "gru"],
        help="Type of RNN cell for sender",
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


def coco_arch_parsing(parser):
    parser.add_argument(
        "--head_choice",
        type=str,
        default="signal_expansion",
        help="Choose the receiver box head module",
        choices=list(HEAD_CHOICES.keys()),
    )

    parser.add_argument(
        "--flat_choice_sender",
        type=str,
        default="AvgPool",
        help="Choose the flat module type",
        choices=list(FLAT_CHOICES.keys()),
    )

    parser.add_argument(
        "--flat_choice_receiver",
        type=str,
        default="AvgPool",
        help="Choose the flat module type",
        choices=list(FLAT_CHOICES.keys()),
    )

    parser.add_argument(
        "--box_head_hidden",
        type=int,
        default=32,
        help="Size of the hidden layer of Receiver (default: 10)",
    )


def training_parsing(parser):
    parser.add_argument(
        "--resume_training",
        type=str2bool,
        nargs="?",
        const=True,
        help="Resume training loading models from '--checkpoint_dir'",
    )

    parser.add_argument(
        "--sender_pretrain",
        type=str,
        nargs="?",
        const=True,
        default="",
        help="Path for pretrained sender weights",
    )

    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.8,
        help="Decay rate for lr ",
    )


def parse_arguments(params=None):
    parser = argparse.ArgumentParser()

    # populate the parser with different options
    training_parsing(parser)
    loss_parser(parser)
    coco_arch_parsing(parser)
    image_parser(parser)
    logs_parser(parser)
    dataloader_parsing(parser)
    egg_arch_parsing(parser)

    # add core opt and print
    opt = core.init(parser, params=params)

    if opt.checkpoint_dir is None:
        opt.checkpoint_dir = "checkpoints"

    if opt.sender_receiver_hidden is not None:
        opt.sender_hidden = opt.sender_receiver_hidden
        opt.receiver_hidden = opt.sender_receiver_hidden

    console.log(sorted(vars(opt).items()))

    if opt.use_rich_traceback:
        from rich.traceback import install

        install()

    # assert the number of classes is less than 90-skip_first
    assert (
        opt.num_classes + opt.skip_first <= 80
    ), f"The number of classes plus the skip must be less than 90, currently {opt.num_classes + opt.skip_first} "

    assert opt.image_resize >= 224, "The size of the image must be minimum 224"
    return opt
