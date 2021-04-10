import csv
from typing import Dict

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.utils import (
    add_row,
    console,
    path_parser,
)


def get_infos(lines: list) -> Dict:
    """
    Estimate the per class accuracy based on the sample found in the interactions.csv
    The names follow this convension;
    X_target Y distr: where X is correct orr wrong based on the prediction and Y is == or != if the target is the same as the distractor.
    """
    accuracy_dict = {}

    def empty_dict():

        return {
            Tot: 0,
            Crr: 0,
            CTED: 0,
            WTED: 0,
            CTND: 0,
            WTND: 0,
            OCL: 0,
            TF: 0,
        }

    for l in lines:
        pred_class = l[2]
        true_class = l[3]
        correct = l[4]
        distract = l[5]
        other_classes = l[6]

        if pred_class not in accuracy_dict:
            accuracy_dict[pred_class] = empty_dict()
        if true_class not in accuracy_dict:
            accuracy_dict[true_class] = empty_dict()
        if distract not in accuracy_dict:
            accuracy_dict[distract] = empty_dict()

        accuracy_dict[pred_class][Tot] += 1
        accuracy_dict[pred_class][OCL] += len(other_classes.split(";"))

        accuracy_dict[pred_class][TF] += 1

        if eval(correct):
            accuracy_dict[pred_class][Crr] += 1

            if distract == true_class:
                accuracy_dict[pred_class][CTED] += 1
            else:
                accuracy_dict[pred_class][CTND] += 1
        else:
            if distract == true_class:
                accuracy_dict[pred_class][WTED] += 1
            else:
                accuracy_dict[pred_class][WTND] += 1

    infos = pd.DataFrame.from_dict(accuracy_dict)

    # normalize number of other classes len
    infos.loc[OCL, :] = (
            infos.loc[OCL, :] / infos.loc[Tot, :]
    )

    to_add = infos.loc[Crr, :] / infos.loc[Tot, :]
    infos = add_row(to_add, Acc, infos)

    to_add = infos.loc[CTED, :] / (
            infos.loc[WTED, :] + infos.loc[CTED, :]
    )
    infos = add_row(to_add, PSC, infos)

    to_add = infos.loc[CTND, :] / (
            infos.loc[WTND, :] + infos.loc[CTND, :]
    )
    infos = add_row(to_add, POC, infos)

    to_add = infos.loc[Tot, :] / sum(infos.loc[Tot, :])
    infos = add_row(to_add, "frequency", infos)

    to_add = infos.loc[CTED, :] + infos.loc[WTED, :]
    to_add /= infos.loc[Tot, :]
    infos = add_row(to_add, ARt, infos)

    infos = infos.fillna(0)

    return infos


def coccurence(lines, classes):
    df = pd.DataFrame(index=classes, columns=classes)
    df = df.fillna(0)

    for l in lines:
        true_class = l[3]
        distract = l[5]

        df[true_class][distract] += 1

    return df


def analysis_df(infos):
    analysis = pd.DataFrame()

    to_add = infos.loc[Acc, :].sum() / infos.shape[1]
    analysis = add_row(to_add, "Total Accuracy", analysis)

    to_add = (
            infos.loc[PSC, :] - infos.loc[POC, :].sum() / infos.shape[1]
    )
    to_add = to_add.mean()
    analysis = add_row(to_add, "Precision difference sc/oc", analysis)

    to_add = infos.loc[Acc, :].corr(infos.loc[Frq, :])
    analysis = add_row(to_add, f"Corr {Acc}-{Frq}", analysis)

    to_add = infos.loc[OCL, :].corr(infos.loc[Frq, :])
    analysis = add_row(to_add, f"Corr {OCL}-{Frq}", analysis)

    to_add = infos.loc[ARt, :].corr(infos.loc[Frq, :])
    analysis = add_row(to_add, f"Corr {ARt}-{Frq}", analysis)

    to_add = infos.loc[ARt, :].corr(infos.loc[Acc, :])
    analysis = add_row(to_add, f"Corr {ARt}-{Acc}", analysis)

    to_add = infos.loc[ARt, :].corr(infos.loc[OCL, :])
    analysis = add_row(to_add, f"Corr {ARt}-{OCL}", analysis)

    return analysis


def accuracy_analysis(interaction_path, out_dir):
    console.log("Computing accuracy analysis...")

    with open(interaction_path, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)

    header = lines.pop(0)
    assert len(lines) > 0, "Empty Csv File!"
    infos = get_infos(lines)

    cooc = coccurence(lines, infos.columns)
    analysis = analysis_df(infos)

    cooc_path = out_dir.joinpath("acc_class_cooc.csv")
    infos_path = out_dir.joinpath("acc_class_infos.csv")
    analysis_path = out_dir.joinpath("acc_analysis.csv")

    cooc.to_csv(cooc_path)
    infos.to_csv(infos_path)
    analysis.to_csv(analysis_path)

    print(f"Out saved in {out_dir}")

    return dict(
        acc_class_cooc=cooc,
        acc_class_infos=infos,
        acc_analysis=analysis,
    )


if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    accuracy_analysis(interaction_path, out_dir)
