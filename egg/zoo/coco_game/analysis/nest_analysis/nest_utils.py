import argparse
import pathlib


def path_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nest_out_path",
        default="/home/dizzi/Downloads/latest_data/",
        type=str,
    )
    p = parser.parse_args()

    return pathlib.Path(p.nest_out_path)
