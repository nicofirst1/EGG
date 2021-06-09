import argparse
import uuid

from egg import core
from egg.zoo.coco_game.archs import HEAD_CHOICES
from egg.zoo.coco_game.utils.utils import console, str2bool


def image_parser(parser):
    parser.add_argument(
        "--image_type",
        type=str,
        default="ImgTargetContext",
        help="Choose the sender input type, an image of the"
             "1) ImgTarget: just the single object in the image"
             "2) ImgContext: the whole image without specifying which objects (target) you're referring to"
             "3) ImgTargetContext: both the target and the whole image",
        choices=["ImgTarget", "ImgContext", "ImgTargetContext"],
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
        "--use_tensorboard_logger",
        type=str2bool,
        nargs="?",
        default=False,
        help="If true, use the tensorboard logger with path 'tensorboard_path",
    )

    parser.add_argument(
        "--use_rich_traceback",
        type=str2bool,
        nargs="?",
        default=False,
        help="If to use the traceback provided by the rich library",
    )

    parser.add_argument(
        "--use_progress_bar",
        type=str2bool,
        nargs="?",
        default=True,
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
        "--train_logging_step",
        type=int,
        default=20,
        help="Number of steps (in batches) before logging during training ",
    )

    parser.add_argument(
        "--val_logging_step",
        type=int,
        default=2,
        help="Number of steps (in batches) before logging during validation",
    )



def dataloader_parsing(parser):
    parser.add_argument(
        "--use_invert_data",
        type=str2bool,
        nargs="?",
        default=False,
        help="Use train as val and val as train",
    )

    parser.add_argument(
        "--use_train_split",
        type=str2bool,
        nargs="?",
        default=False,
        help="Split train data into train/val",
    )

    parser.add_argument(
        "--use_dummy_val",
        type=str2bool,
        nargs="?",
        default=False,
        help="Use dummy data for validation",
    )

    parser.add_argument(
        "--use_train_val",
        type=str2bool,
        nargs="?",
        default=False,
        help="Use train data for validation",
    )

    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Random seed for dataloaders",
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
        default="feature_reduction",
        help="Choose the receiver box head module",
        choices=list(HEAD_CHOICES.keys()),
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
        const=False,
        help="Resume training loading models from '--checkpoint_dir'",
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
    coco_arch_parsing(parser)
    image_parser(parser)
    logs_parser(parser)
    dataloader_parsing(parser)
    egg_arch_parsing(parser)

    # add core opt and print
    opt = core.init(parser, params=params)

    if opt.sender_receiver_hidden is not None:
        opt.sender_hidden = opt.sender_receiver_hidden
        opt.receiver_hidden = opt.sender_receiver_hidden

    console.log(sorted(vars(opt).items()))

    if opt.use_rich_traceback:
        from rich.traceback import install

        install()

    assert opt.image_resize >= 224, "The size of the image must be minimum 224"
    return opt
