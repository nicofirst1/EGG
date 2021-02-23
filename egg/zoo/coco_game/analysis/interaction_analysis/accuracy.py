import csv
from typing import Dict

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis.utils import path_parser


def get_infos(lines: list) -> Dict:
    """
    Estimate the per class accuracy based on the sample found in the interactions.csv
    """
    accuracy_dict = {}

    for l in lines:
        pred_class = l[2]
        true_class = l[3]
        correct = l[4]
        distract = l[5]
        other_classes = l[6]

        if pred_class not in accuracy_dict:
            accuracy_dict[pred_class] = dict(
                total=0,
                true=0,
                true_sc=0,
                false_sc=0,
                true_oc=0,
                false_oc=0,
                other_classes_len=0,
            )

        accuracy_dict[pred_class]["total"] += 1
        accuracy_dict[pred_class]["other_classes_len"] += len(other_classes.split(";"))
        if eval(correct):
            accuracy_dict[pred_class]["true"] += 1

            if distract == true_class:
                accuracy_dict[pred_class]["true_sc"] += 1
            else:
                accuracy_dict[pred_class]["true_oc"] += 1
        else:
            if distract == true_class:
                accuracy_dict[pred_class]["false_sc"] += 1
            else:
                accuracy_dict[pred_class]["false_oc"] += 1

    infos = pd.DataFrame.from_dict(accuracy_dict)

    # normalize number of other classes len
    infos.loc["other_classes_len", :] = (
        infos.loc["other_classes_len", :] / infos.loc["total", :]
    )

    to_add = infos.loc["true", :] / infos.loc["total", :]
    to_add.name = "accuracy"
    infos = infos.append(to_add)

    to_add = infos.loc["true_sc", :] / (
        infos.loc["false_sc", :] + infos.loc["true_sc", :]
    )
    to_add.name = "prec_sc"
    infos = infos.append(to_add)

    to_add = infos.loc["true_oc", :] / (
        infos.loc["false_oc", :] + infos.loc["true_oc", :]
    )
    to_add.name = "prec_oc"
    infos = infos.append(to_add)

    to_add = infos.loc["total", :] / sum(infos.loc["total", :])
    to_add.name = "frequency"
    infos = infos.append(to_add)

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


def accuracy_analysis(interaction_path, out_dir):

    with open(interaction_path, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)

    header = lines.pop(0)
    infos = get_infos(lines)

    total_accuracy = sum(infos.loc["accuracy", :]) / infos.shape[1]
    prec_diff = sum(infos.loc["prec_sc", :] - infos.loc["prec_oc", :]) / infos.shape[1]
    acc_freq_corr = infos.loc["accuracy", :].corr(infos.loc["frequency", :])
    acc_oc_len_corr = infos.loc["other_classes_len", :].corr(infos.loc["frequency", :])

    cooc = coccurence(lines, infos.columns)

    cooc_path = out_dir.joinpath("co_occurence.csv")
    infos_path = out_dir.joinpath("infos.csv")
    analysis_path = out_dir.joinpath("analysis.csv")

    cooc.to_csv(cooc_path)
    infos.to_csv(infos_path)

    with open(analysis_path, "w+") as f:
        head = [
            "Total Accuracy",
            "Correlation Accuracy Frequency",
            "Correlation Accuracy Other classes",
            "Precision difference sc/oc",
        ]
        f.write(",".join(head))
        f.write("\n")
        vals = [total_accuracy, acc_freq_corr, acc_oc_len_corr, prec_diff]
        vals = [str(x) for x in vals]
        f.write(",".join(vals))
        f.write("\n")

    print(f"Out saved in {out_dir}")
