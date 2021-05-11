import csv
import json
import pickle
from typing import Dict

import pandas as pd
from rich.progress import track

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.utils import console, path_parser, split_line


def get_infos(lines: list, max_len) -> Dict:
    """
    Estimate the frequency of symbols and sequences normalized by the total number of messages
    """
    infos_class = {
        "message_len": 0,
        Sy: {},
        Se: {},
    }
    infos_superclass = {
        "message_len": 0,
        Sy: {},
        Se: {},
    }

    classes = []
    superclasses = []

    for l in lines:
        data = split_line(l)
        message = data['message']
        target_class = data['target_class']
        distract_class = data['distractor_class']
        target_superclass = data['target_superclass']
        distract_superclass = data['distractor_superclass']

        if target_class not in classes:
            classes.append(target_class)
        if distract_class not in classes:
            classes.append(distract_class)

        if target_superclass not in superclasses:
            superclasses.append(target_superclass)
        if distract_superclass not in superclasses:
            superclasses.append(distract_superclass)

        msg_len = message.index("0")
        infos_class["message_len"] += msg_len
        infos_superclass["message_len"] += msg_len

        for m in message[:max_len]:

            if m not in infos_class[Sy].keys():
                infos_class[Sy][m] = 0
            infos_class[Sy][m] += 1

            if m not in infos_superclass[Sy].keys():
                infos_superclass[Sy][m] = 0
            infos_superclass[Sy][m] += 1

        symbol_seq = "".join(message[:msg_len])
        if symbol_seq == "":
            symbol_seq = "0"

        if symbol_seq not in infos_class[Se].keys():
            infos_class[Se][symbol_seq] = 0
        if symbol_seq not in infos_superclass[Se].keys():
            infos_superclass[Se][symbol_seq] = 0

        infos_class[Se][symbol_seq] += 1
        infos_superclass[Se][symbol_seq] += 1

    lines_len = len(lines)
    infos_class["message_len"] /= lines_len
    #infos_class[Se] = {k: v / lines_len for k, v in infos_class[Se].items()}

    infos_superclass["message_len"] /= lines_len
    #infos_superclass[Se] = {k: v / lines_len for k, v in infos_superclass[Se].items()}

    # for k in infos_class[Sy].keys():
    #     infos_class[Sy][k] /= lines_len
    #     infos_class[Sy][k] /= max_len

    # for k in infos_superclass[Sy].keys():
    #     infos_superclass[Sy][k] /= lines_len
    #     infos_superclass[Sy][k] /= max_len

    return (infos_class, classes), (infos_superclass, superclasses)


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

        # df /= df.sum().sum()
        tensor[true_c] = df

    return tensor


def ambiguity_richness(lang_sequence_cooc: Dict) -> Dict:
    ar_res = {}
    ar_perc_res = {}
    for trg, v in lang_sequence_cooc.items():
        total = 0
        total_perc = 0
        ar_res[trg] = 0
        ar_perc_res[trg] = 0

        for dstr, v2 in v.items():
            if dstr == trg:
                ar_res[trg] += len(v2)
                ar_perc_res[trg] += sum(v2.values())

            total += len(v2)
            total_perc += sum(v2.values())
        #
        # try:
        #     ar_res[trg] /= total
        #     ar_perc_res[trg] /= total_perc
        # except ZeroDivisionError:
        #     pass

    return ar_res, ar_perc_res


def cooc_tensor(lines, symbols, sequences, classes, superclasses, max_len):
    """
    Return the number of times a sequence or a symbol is used when a particular pair of target/distractors is in place
    """
    class_symbol_df = pd.DataFrame(index=classes, columns=symbols)
    class_symbol_df = class_symbol_df.fillna(0)

    class_sequence_df = pd.DataFrame(index=classes, columns=sequences)
    class_sequence_df = class_sequence_df.fillna(0)

    superclass_symbol_df = pd.DataFrame(index=superclasses, columns=symbols)
    superclass_symbol_df = superclass_symbol_df.fillna(0)

    superclass_sequence_df = pd.DataFrame(index=superclasses, columns=sequences)
    superclass_sequence_df = superclass_sequence_df.fillna(0)

    class_tensor = {k: {} for k in classes}
    superclass_tensor = {k: {} for k in superclasses}

    for l in track(lines, description="Computing language analysis..."):
        data = split_line(l)
        message = data['message']
        target_class = data['target_class']
        distractor_class = data['distractor_class']

        target_superclass = data['target_superclass']
        distractor_superclass = data['distractor_superclass']

        msg_len = message.index("0")
        for m in message[:max_len]:
            class_symbol_df[m][target_class] += 1
            superclass_symbol_df[m][target_superclass] += 1

        symbol_seq = "".join(message[:msg_len])

        if symbol_seq == "":
            symbol_seq = "0"

        class_sequence_df[symbol_seq][target_class] += 1
        superclass_sequence_df[symbol_seq][target_superclass] += 1

        if distractor_class not in class_tensor[target_class].keys():
            class_tensor[target_class][distractor_class] = {}

        if distractor_superclass not in superclass_tensor[target_superclass].keys():
            superclass_tensor[target_superclass][distractor_superclass] = {}

        if symbol_seq not in class_tensor[target_class][distractor_class].keys():
            class_tensor[target_class][distractor_class][symbol_seq] = 0

        if symbol_seq not in superclass_tensor[target_superclass][distractor_superclass].keys():
            superclass_tensor[target_superclass][distractor_superclass][symbol_seq] = 0

        class_tensor[target_class][distractor_class][symbol_seq] += 1
        superclass_tensor[target_superclass][distractor_superclass][symbol_seq] += 1

    # class_symbol_df /= len(lines)
    # class_symbol_df /= max_len
    # class_sequence_df /= len(lines)
    #
    # superclass_symbol_df /= len(lines)
    # superclass_symbol_df /= max_len
    # superclass_sequence_df /= len(lines)

    classes = (class_symbol_df, class_sequence_df, class_tensor)
    superclasses = (superclass_symbol_df, superclass_sequence_df, superclass_tensor)

    return classes, superclasses


def add_rows(symbol_df, infos, sequence_df):
    to_add = symbol_df.sum(axis=1)
    symbol_df[CR] = to_add

    to_add = pd.Series(infos[Sy])
    to_add.name = Frq
    symbol_df = symbol_df.append(to_add)

    to_add = sequence_df.sum(axis=1)
    sequence_df[CR] = to_add

    to_add = pd.Series(infos[Se])
    to_add.name = Frq
    sequence_df = sequence_df.append(to_add)

    return symbol_df, sequence_df


def language_analysis(interaction_path, out_dir):
    with open(interaction_path, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)

    header = lines.pop(0)

    max_len = lines[0][1].split(";")
    max_len = len(max_len) - 1

    classes, superclasses = get_infos(lines, max_len)

    class_info, classes = classes
    superclass_info, superclasses = superclasses

    symbols = class_info[Sy].keys()
    sequences = class_info[Se].keys()

    classes, superclasses = cooc_tensor(
        lines, symbols, sequences, classes, superclasses, max_len
    )

    class_symbol_df, class_sequence_df, class_tensor_cooc = classes
    superclass_symbol_df, superclass_sequence_df, superclass_tensor_cooc = superclasses

    class_tensor = language_tensor(class_tensor_cooc)
    superclass_tensor = language_tensor(superclass_tensor_cooc)

    class_symbol_df, class_sequence_df = add_rows(class_symbol_df, class_info, class_sequence_df)
    superclass_symbol_df, superclass_sequence_df = add_rows(superclass_symbol_df, superclass_info,
                                                            superclass_sequence_df)

    class_symbol_path = out_dir.joinpath("lang_class_symbol.csv")
    class_sequence_cooc_path = out_dir.joinpath("lang_class_sequence_cooc.json")
    class_sequence_path = out_dir.joinpath("lang_class_sequence.csv")
    class_tensor_path = out_dir.joinpath("lang_class_tensor.pkl")

    superclass_symbol_path = out_dir.joinpath("lang_superclass_symbol.csv")
    superclass_sequence_cooc_path = out_dir.joinpath("lang_superclass_sequence_cooc.json")
    superclass_sequence_path = out_dir.joinpath("lang_superclass_sequence.csv")
    superclass_tensor_path = out_dir.joinpath("lang_superclass_tensor.pkl")

    class_symbol_df.to_csv(class_symbol_path)
    class_sequence_df.to_csv(class_sequence_path)

    superclass_symbol_df.to_csv(superclass_symbol_path)
    superclass_sequence_df.to_csv(superclass_sequence_path)

    with open(class_sequence_cooc_path, "w") as f:
        json.dump(class_tensor_cooc, f)

    with open(superclass_sequence_cooc_path, "w") as f:
        json.dump(superclass_tensor_cooc, f)

    with open(class_tensor_path, "wb") as f:
        pickle.dump(class_tensor, f)

    with open(superclass_tensor_path, "wb") as f:
        pickle.dump(superclass_tensor, f)

    console.log(f"Files save in {out_dir}")

    return dict(
        lang_class_symbol=class_symbol_df,
        lang_class_sequence=class_sequence_df,
        lang_class_sequence_cooc=class_tensor_cooc,
        lang_class_tensor=class_tensor,
        lang_superclass_symbol=superclass_symbol_df,
        lang_superclass_sequence=superclass_sequence_df,
        lang_superclass_sequence_cooc=superclass_tensor_cooc,
        lang_superclass_tensor=superclass_tensor,
    )


if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    language_analysis(interaction_path, out_dir)
