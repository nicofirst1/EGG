from termcolor import colored

from egg.zoo.coco_game.main import main

base_args = [
    "--batch_size",
    "32",
    "--max_len",
    "5",
    "--train_data_perc",
    "0.0005",
    "--test_data_perc",
    "0.01",
    "--num_workers",
    "0",
    "--n_epochs",
    "1",
    "--no_cuda",
    "--checkpoint_dir",
    "checkpoints",
    "--train_logging_step",
    "3",
    "--test_logging_step",
    "2",
]


def iou_loss(args):
    args += ["--loss_type", "iou"]
    main(params=args)


def giou_loss(args):
    args += ["--loss_type", "giou"]
    main(params=args)


def diou_loss(args):
    args += ["--loss_type", "diou"]
    main(params=args)


def ciou_loss(args):
    args += ["--loss_type", "ciou"]
    main(params=args)


def only_iou_loss(args):
    args += [
        "--l1_lambda",
        "0",
        "--cross_lambda",
        "0",
        "--iou_lambda",
        "1",
        "--n_epochs",
        "5",
        "--log_dir_uid",
        "iou",
    ]
    main(params=args)


def only_l1_loss(args):
    args += [
        "--l1_lambda",
        "1",
        "--cross_lambda",
        "0",
        "--iou_lambda",
        "0",
        "--n_epochs",
        "5",
        "--log_dir_uid",
        "l1",
    ]
    main(params=args)


def only_cross_loss(args):
    args += [
        "--l1_lambda",
        "0",
        "--cross_lambda",
        "1",
        "--iou_lambda",
        "0",
        "--n_epochs",
        "5",
        "--log_dir_uid",
        "cross",
    ]
    main(params=args)


def test_loss_mods():
    print(colored(f"Testing only iou loss...", "yellow"))
    only_iou_loss(base_args)

    print(colored(f"Testing only l1 loss...", "yellow"))
    only_l1_loss(base_args)

    print(colored(f"Testing only cross loss...", "yellow"))
    only_cross_loss(base_args)

    exit()

    print(colored(f"Testing iou loss...", "yellow"))
    iou_loss(base_args)

    print(colored(f"\n\nTesting giou loss...", "yellow"))
    giou_loss(base_args)

    print(colored(f"\n\nTesting diou loss...", "yellow"))
    diou_loss(base_args)

    print(colored(f"\n\nTesting ciou loss...", "yellow"))
    ciou_loss(base_args)

    print(colored(f"\n\nDone testing losses", "yellow"))


if __name__ == "__main__":
    test_loss_mods()
