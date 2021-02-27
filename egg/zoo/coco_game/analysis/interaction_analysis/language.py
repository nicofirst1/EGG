import csv
import json
import pickle
from typing import Dict

import pandas as pd
from rich.progress import track

from egg.zoo.coco_game.analysis.interaction_analysis.utils import path_parser, console


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
        distract = l[5]

        if true_class not in classes:
            classes.append(true_class)
        if distract not in classes:
            classes.append(distract)

        msg_len = message.index("0")
        infos["message_len"] += msg_len

        for m in message[:max_len]:

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


def language_tensor(lang_sequence_cooc):
    tensor = {}
    for true_c, distr_dict in lang_sequence_cooc.items():
        sequences = [list(x.keys()) for x in distr_dict.values()]
        sequences = [x for sub in sequences for x in sub]
        sequences = set(sequences)

        distractors = list(distr_dict.keys())

        df = pd.DataFrame(index=sequences, columns=distractors).fillna(0)

        for dist_k, seqs in distr_dict.items():

            for seq_k, seq_v in seqs.items():
                df[dist_k][seq_k] += seq_v

        tensor[true_c] = df

    return tensor


def ambiguity_richness(lang_sequence_cooc: Dict) -> Dict:
    ar_res = {}
    ar_perc_res = {}
    for k, v in lang_sequence_cooc.items():
        total = 0
        total_perc = 0
        ar_res[k] = 0
        ar_perc_res[k] = 0

        for k2, v2 in v.items():
            if k2 == k:
                ar_res[k] += len(v2)
                ar_perc_res[k] += sum(v2.values())
            else:
                total += len(v2)
                total_perc += sum(v2.values())

        ar_res[k] /= total
        ar_perc_res[k] /= total_perc

    return ar_res, ar_perc_res


def coccurence(lines, symbols, sequences, classes, max_len):
    symbol_df = pd.DataFrame(index=classes, columns=symbols)
    symbol_df = symbol_df.fillna(0)

    sequence_df = pd.DataFrame(index=classes, columns=sequences)
    sequence_df = sequence_df.fillna(0)

    sequence_cooc_tensor = {k: {} for k in classes}

    for l in track(lines, description="Computing language analysis..."):
        message = l[1].split(";")
        true_class = l[3]
        pred_class = l[2]
        correct = l[4]
        distract = l[5]
        other_classes = l[6]

        msg_len = message.index("0")
        for m in message[:max_len]:
            symbol_df[m][true_class] += 1

        symbol_seq = "".join(message[:msg_len])
        sequence_df[symbol_seq][true_class] += 1

        if distract not in sequence_cooc_tensor[true_class].keys():
            sequence_cooc_tensor[true_class][distract] = {}

        if symbol_seq not in sequence_cooc_tensor[true_class][distract].keys():
            sequence_cooc_tensor[true_class][distract][symbol_seq] = 0

        sequence_cooc_tensor[true_class][distract][symbol_seq] += 1

    symbol_df /= len(lines)
    symbol_df /= max_len
    sequence_df /= len(lines)

    return symbol_df, sequence_df, sequence_cooc_tensor


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
    symbol_df, sequence_df, sequence_cooc_tensor = coccurence(
        lines, symbols, sequences, classes, max_len
    )

    tensor=language_tensor(sequence_cooc_tensor)

    to_add = symbol_df.sum(axis=1)
    symbol_df["class_richness"] = to_add

    to_add = pd.Series(infos["symbols"])
    to_add.name = "frequency"
    symbol_df = symbol_df.append(to_add)

    to_add = sequence_df.sum(axis=1)
    sequence_df["class_richness"] = to_add

    to_add = pd.Series(infos["sequences"])
    to_add.name = "frequency"
    sequence_df = sequence_df.append(to_add)

    symbol_path = out_dir.joinpath("lang_symbol.csv")
    sequence_cooc_path = out_dir.joinpath("lang_sequence_cooc.json")
    sequence_path = out_dir.joinpath("lang_sequence.csv")
    tensor_path = out_dir.joinpath("lang_tensor.pkl")

    symbol_df.to_csv(symbol_path)
    sequence_df.to_csv(sequence_path)

    with open(sequence_cooc_path, "w") as f:
        json.dump(sequence_cooc_tensor, f)

    with open(tensor_path, "wb") as f:
        pickle.dump(tensor, f)

    console.log(f"Files save in {out_dir}")

    return dict(
        lang_symbol=symbol_df,
        lang_sequence=sequence_df,
        lang_sequence_cooc=sequence_cooc_tensor,
        lang_tensor=tensor,
    )


if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    language_analysis(interaction_path, out_dir)
