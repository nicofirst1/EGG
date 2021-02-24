import argparse
import pathlib
import pandas as pd

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

    s=pd.Series(row_value)
    s.name=row_name

    return df.append(s)