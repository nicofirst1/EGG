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
]


def single_receiver(args):
    args += ["--head_choice", "single"]
    main(params=args)


def multi_receiver(args):
    args += ["--head_choice", "multi"]
    main(params=args)


def yolo_receiver(args):
    args += ["--head_choice", "yolo"]
    main(params=args)


def test_receiver_mods():
    print(colored(f"Testing single receiver...", "yellow"))
    single_receiver(base_args)
    print(colored(f"Testing multi receiver...", "yellow"))
    multi_receiver(base_args)
    print(colored(f"Testing yolo receiver...", "yellow"))
    yolo_receiver(base_args)

    print(colored(f"\n\nDone testing receiver mods", "yellow"))


if __name__ == "__main__":
    test_receiver_mods()
