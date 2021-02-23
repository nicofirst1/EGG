from termcolor import colored

from egg.zoo.coco_game.train import main

base_args = [
    "--batch_size",
    "64",
    "--max_len",
    "5",
    "--train_data_perc",
    "0.001",
    "--test_data_perc",
    "0.01",
    "--num_workers",
    "0",
    "--n_epochs",
    "1",
    "--lr",
    "0.0001",
    "--image_save_eval",
    "20",
    "--train_log_prob",
    "0.01",
    "--test_log_prob",
    "0.01",
    "--checkpoint_dir",
    "./checkpoints",
    "--checkpoint_freq",
    "0",
    "--tensorboard",
]


def seg_sender(args):
    args += ["--image_type", "seg"]
    main(params=args)


def img_sender(args):
    args += ["--image_type", "img"]
    main(params=args)


def both_mul_sender(args):
    args += ["--image_type", "both", "--image_union", "mul"]
    main(params=args)


def both_cat_sender(args):
    args += ["--image_type", "both", "--image_union", "cat"]
    main(params=args)


def test_sender_mods():
    print(colored(f"Testing sender with segment input...", "yellow"))
    seg_sender(base_args)

    print(colored(f"\n\nTesting sender with image input...", "yellow"))
    img_sender(base_args)

    print(colored(f"\n\nTesting sender with both input using mul...", "yellow"))
    both_mul_sender(base_args)

    print(colored(f"\n\nTesting sender with both input using cat...", "yellow"))
    both_cat_sender(base_args)

    print(colored(f"\n\nDone testing sender mods", "yellow"))


if __name__ == "__main__":
    test_sender_mods()
