import argparse
import pathlib

import pandas as pd
from rich.console import Console

console = Console()


def path_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interaction_path",
        default="/home/dizzi/Desktop/EGG/egg/zoo/coco_game/Logs/test/runs/interactions.csv",
        type=str,
    )

    parser.add_argument(
        "--out_dir",
        default="",
        type=str,
    )

    p = parser.parse_args()

    interaction_path = pathlib.Path(p.interaction_path)

    if not p.out_dir:
        p.out_dir = interaction_path.parent.joinpath("Analysis_out")

    out_dir = pathlib.Path(p.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return interaction_path, out_dir


def add_row(row_value, row_name, df: pd.DataFrame):
    s = pd.Series(row_value)
    s.name = row_name

    return df.append(s)


def split_line(line):
    """
    Parse the line coming from the csv into a dict
    """
    data = {}

    data['epoch'] = line[0]
    data['message'] = line[1].split(";")
    data['predicted'] = line[2]
    data['target'] = line[3]
    data['is_correct'] = line[4]
    data['distractor'] = line[5]
    data['other_classes'] = line[6]

    return data
