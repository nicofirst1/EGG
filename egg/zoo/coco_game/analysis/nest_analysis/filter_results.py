import argparse
import pathlib

import pandas as pd

# ADD COLUMNS HERE
columns_of_interests = [
    "head_choice",
    "image_union",
    "max_len",
    "vocab_size",
    "sender_receiver_hidden",
    "train_epoch",
    "train_accuracy_receiver",
    "test_accuracy_receiver",
]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--csv_path",
    default="/home/dizzi/Downloads/best_temp/results_best_temp.csv",
    type=str,
)

parser.add_argument(
    "--out_file_path",
    default="",
    type=str,
)
p = parser.parse_args()

csv_path = pathlib.Path(p.csv_path)

if p.out_file_path == "":
    p.out_file_path = csv_path.parent

out_path = pathlib.Path(p.out_file_path)
out_path = out_path.joinpath(f"{csv_path.stem}_filtered.csv")

columns_of_interests = set(columns_of_interests)
df = pd.read_csv(csv_path, index_col=0)

df_cols = set(df.columns)
to_drop = df_cols - columns_of_interests

df = df.drop(to_drop, axis=1)
df.to_csv(out_path)
print(f"File save in {out_path}")
