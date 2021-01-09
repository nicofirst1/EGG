from termcolor import colored

from egg.zoo.coco_game.main import main

base_args = [
    "--batch_size",
    "32",
    "--max_len",
    "5",
    "--train_data_perc",
    "0.001",
    "--test_data_perc",
    "0.02",
    "--num_workers",
    "0",
    "--no_cuda",
    "--n_epochs",
    "3",
    "--checkpoint_dir",
    "./checkpoints",
    "--checkpoint_freq",
    "1",
    "--tensorboard",
    "--train_logging_step",
    "3",
    "--test_logging_step",
    "1",
]


def max_log_prob(args):
    args += ["--train_log_prob", "1"]
    args += ["--test_log_prob", "1"]
    main(params=args)


def min_log_prob(args):
    args += ["--train_log_prob", "0"]
    args += ["--test_log_prob", "0"]
    main(params=args)


def mean_log_prob(args):
    args += ["--train_log_prob", "0.5"]
    args += ["--test_log_prob", "0.5"]
    main(params=args)


def multi_worker(args):
    args += ["--num_workers", "4"]
    main(params=args)


def defored_input(args):
    args += ["--deform_rec_input"]
    main(params=args)


def num_classes(args):
    try:
        new_args = args + ["--num_classes", "80", "--skip_first", "20"]
        main(params=new_args)
        raise Exception("Did not fail!")
    except AssertionError:
        # should fail
        pass

    new_args = args + ["--num_classes", "10", "--skip_first", "20"]
    main(params=new_args)

    new_args = args + ["--num_classes", "10", "--skip_first", "0"]
    main(params=new_args)


def test_data_mods():
    print(colored(f"\n\nTesting number of classes...", "yellow"))
    num_classes(base_args)

    print(colored(f"\n\nTesting max log prob...", "yellow"))
    max_log_prob(base_args)

    print(colored(f"Testing multiple workers...", "yellow"))
    multi_worker(base_args)

    print(colored(f"\n\nTesting min log prob...", "yellow"))
    min_log_prob(base_args)

    print(colored(f"\n\nTesting mean log prob...", "yellow"))
    mean_log_prob(base_args)

    print(colored(f"\n\nTesting deformed input...", "yellow"))
    defored_input(base_args)

    print(colored(f"\n\nDone testing data", "yellow"))


if __name__ == "__main__":
    test_data_mods()
