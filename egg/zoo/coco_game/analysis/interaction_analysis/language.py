import csv
from typing import Dict

import pandas as pd


def get_infos(lines: list, max_len) -> Dict:
    """
    Estimate the per class accuracy based on the sample found in the interactions.csv
    """
    infos = dict(
        message_len=0,
        symbols={},
        sequences={},
    )

    classes = []

    for l in lines:
        message = l[1].split(";")
        true_class = l[3]

        if true_class not in classes:
            classes.append(true_class)

        msg_len = message.index("0")
        infos["message_len"] += msg_len

        for m in message[:msg_len]:

            if m not in infos["symbols"].keys():
                infos["symbols"][m] = 0
            infos["symbols"][m] += 1

        symbol_seq = "".join(message[:msg_len])
        if symbol_seq not in infos["sequences"].keys():
            infos["sequences"][symbol_seq] = 0

        infos["sequences"][symbol_seq] += 1

    lines_len = len(lines)
    infos["message_len"] /= lines_len
    infos["sequences"] = {k: v / lines_len for k, v in infos["sequences"].items()}

    for k in infos["symbols"].keys():
        infos["symbols"][k] /= lines_len
        infos["symbols"][k] /= max_len

    return infos, classes


def coccurence(lines, symbols, sequences, classes, max_len):
    symbol_df = pd.DataFrame(index=classes, columns=symbols)
    symbol_df = symbol_df.fillna(0)

    sequence_df = pd.DataFrame(index=classes, columns=sequences)
    sequence_df = sequence_df.fillna(0)

    for l in lines:
        message = l[1].split(";")
        true_class = l[3]

        msg_len = message.index("0")
        for m in message[:msg_len]:
            symbol_df[m][true_class] += 1

        symbol_seq = "".join(message[:msg_len])
        sequence_df[symbol_seq][true_class] += 1

    symbol_df /= len(lines)
    symbol_df /= max_len
    sequence_df /= len(lines)

    return symbol_df, sequence_df


def language_analysis(interaction_path, out_dir):
    with open(interaction_path, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)

    header = lines.pop(0)

    max_len = lines[0][1].split(";")
    max_len = len(max_len) - 1

    infos, classes = get_infos(lines, max_len)
    symbols = infos["symbols"].keys()
    sequences = infos["sequences"].keys()
    symbol_df, sequence_df = coccurence(lines, symbols, sequences, classes, max_len)

    symbol_freq = pd.Series(infos["symbols"])
    symbol_freq.name = "frequency"
    symbol_df = symbol_df.append(symbol_freq)

    seq_freq = pd.Series(infos["sequences"])
    seq_freq.name = "frequency"
    sequence_df = sequence_df.append(seq_freq)

    symbol_path = out_dir.joinpath("symbol.csv")
    sequence_path = out_dir.joinpath("sequence.csv")

    symbol_df.to_csv(symbol_path)
    sequence_df.to_csv(sequence_path)

    print(f"Files save in {out_dir}")
