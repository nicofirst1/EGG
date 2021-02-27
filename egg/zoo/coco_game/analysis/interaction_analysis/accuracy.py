import csv
from typing import Dict

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis.utils import add_row, path_parser, console


def get_infos(lines: list) -> Dict:
    """
    Estimate the per class accuracy based on the sample found in the interactions.csv
    """
    accuracy_dict = {}

    def empty_dict():

        return {
            "total": 0,
            "correct": 0,
            "correct_tartget=distr": 0,
            "wrong_target=distr": 0,
            "correct_target!=distr": 0,
            "wrong_target!=distr": 0,
            "other_classes_len": 0,
            "target_freq": 0,
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

        accuracy_dict[pred_class]["total"] += 1
        accuracy_dict[pred_class]["other_classes_len"] += len(other_classes.split(";"))

        accuracy_dict[pred_class]["target_freq"] += 1

        if eval(correct):
            accuracy_dict[pred_class]["correct"] += 1

            if distract == true_class:
                accuracy_dict[pred_class]["correct_tartget=distr"] += 1
            else:
                accuracy_dict[pred_class]["correct_target!=distr"] += 1
        else:
            if distract == true_class:
                accuracy_dict[pred_class]["wrong_target=distr"] += 1
            else:
                accuracy_dict[pred_class]["wrong_target!=distr"] += 1

    infos = pd.DataFrame.from_dict(accuracy_dict)

    # normalize number of other classes len
    infos.loc["other_classes_len", :] = (
        infos.loc["other_classes_len", :] / infos.loc["total", :]
    )

    to_add = infos.loc["correct", :] / infos.loc["total", :]
    infos = add_row(to_add, "accuracy", infos)

    to_add = infos.loc["correct_tartget=distr", :] / (
        infos.loc["wrong_target=distr", :] + infos.loc["correct_tartget=distr", :]
    )
    infos = add_row(to_add, "precision_sc", infos)

    to_add = infos.loc["correct_target!=distr", :] / (
        infos.loc["wrong_target!=distr", :] + infos.loc["correct_target!=distr", :]
    )
    infos = add_row(to_add, "precision_oc", infos)

    to_add = infos.loc["total", :] / sum(infos.loc["total", :])
    infos = add_row(to_add, "frequency", infos)

    to_add = infos.loc["correct_tartget=distr", :] + infos.loc["wrong_target=distr", :]
    to_add /= infos.loc["total", :]
    infos = add_row(to_add, "ambiguity_rate", infos)

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

    to_add = infos.loc["accuracy", :].sum() / infos.shape[1]
    analysis = add_row(to_add, "Total Accuracy", analysis)

    to_add = (
        infos.loc["precision_sc", :]
        - infos.loc["precision_oc", :].sum() / infos.shape[1]
    )
    to_add = to_add.mean()
    analysis = add_row(to_add, "Precision difference sc/oc", analysis)

    to_add = infos.loc["accuracy", :].corr(infos.loc["frequency", :])
    analysis = add_row(to_add, "Corr Accuracy-Frequency", analysis)

    to_add = infos.loc["other_classes_len", :].corr(infos.loc["frequency", :])
    analysis = add_row(to_add, "Corr OtherClassLen-Frequency", analysis)

    to_add = infos.loc["ambiguity_rate", :].corr(infos.loc["frequency", :])
    analysis = add_row(to_add, "Corr AmbiguityRate-Frequency", analysis)

    to_add = infos.loc["ambiguity_rate", :].corr(infos.loc["accuracy", :])
    analysis = add_row(to_add, "Corr AmbiguityRate-Accuracy", analysis)

    to_add = infos.loc["ambiguity_rate", :].corr(infos.loc["other_classes_len", :])
    analysis = add_row(to_add, "Corr AmbiguityRate-OtherClassLen", analysis)

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
