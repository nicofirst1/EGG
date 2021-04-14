import argparse
import json
import pathlib
import pickle

import pandas as pd
from rich.console import Console

console = Console()


def path_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interaction_path",
        default="/home/dizzi/Desktop/EGG/egg/zoo/coco_game/Logs/seg/runs/interactions.csv",
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
    data['pred_class'] = line[2]
    data['pred_superclass'] = line[3]
    data['target_class'] = line[4]
    data['target_superclass'] = line[5]
    data['is_correct'] = line[6]
    data['distractor_class'] = line[7]
    data['distractor_superclass'] = line[8]
    data['other_classes'] = line[9]
    data['other_superclasses'] = line[10]

    return data


def load_generate_files(out_dir, filter):
    filter=f"*_{filter}"
    files = list(out_dir.rglob(f"{filter}*.json"))
    files += list(out_dir.rglob(f"{filter}*.csv"))
    files += list(out_dir.rglob(f"{filter}*.pkl"))
    res_dict = {}

    for path in files:

        if path.suffix == ".csv":
            df = pd.read_csv(path, index_col=0)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                df = json.load(f)

        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                df = pickle.load(f)
        else:
            raise KeyError(f"Unrecognized stem {path.stem}")

        res_dict[path.stem] = df

    return res_dict
