import csv
import json
from typing import Dict

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.utils import (
    add_row,
    console,
    path_parser, split_line,
)


def get_infos(lines: list) -> Dict:
    """
    Estimate the per class accuracy based on the sample found in the interactions.csv
    The names follow this convension;
    X_target Y distr: where X is correct orr wrong based on the prediction and Y is == or != if the target is the same as the distractor.
    """
    accuracy_class_dict = {}
    accuracy_superclass_dict = {}

    def normalize(infos):

        ocl = infos.loc[OCL, :] / infos.loc[OCL, :].sum()
        tot = infos.loc[Tot, :]

        tf = infos.loc[TF]
        infos = infos.drop(index=[OCL, Tot, TF])

        infos /= tf

        tf /= tot
        tot /= tot.sum()

        infos.loc[OCL] = ocl
        infos.loc[TF] = tf
        infos.loc[Tot] = tot

        infos = infos.rename(index={Crr: Acc, Tot: Frq})

        to_add = infos.loc[CTED, :] / (
                infos.loc[WTED, :] + infos.loc[CTED, :]
        )
        infos = add_row(to_add, PSC, infos)

        to_add = infos.loc[CTND, :] / (
                infos.loc[WTND, :] + infos.loc[CTND, :]
        )
        infos = add_row(to_add, POC, infos)

        to_add = infos.loc[CTED, :] + infos.loc[WTED, :]
        infos = add_row(to_add, ARt, infos)

        infos = infos.fillna(0)
        return infos

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

        data = split_line(l)
        pred_class = data['pred_class']
        pred_superclass = data['pred_superclass']
        target_class = data['target_class']
        target_superclass = data['target_superclass']
        correct = data['is_correct']
        distractor_class = data['distractor_class']
        distractor_superclass = data['distractor_superclass']
        other_classes = data['other_classes']
        other_superclasses = data['other_superclasses']

        if pred_class not in accuracy_class_dict:
            accuracy_class_dict[pred_class] = empty_dict()
        if target_class not in accuracy_class_dict:
            accuracy_class_dict[target_class] = empty_dict()
        if distractor_class not in accuracy_class_dict:
            accuracy_class_dict[distractor_class] = empty_dict()

        if pred_superclass not in accuracy_superclass_dict:
            accuracy_superclass_dict[pred_superclass] = empty_dict()
        if target_superclass not in accuracy_superclass_dict:
            accuracy_superclass_dict[target_superclass] = empty_dict()
        if distractor_superclass not in accuracy_superclass_dict:
            accuracy_superclass_dict[distractor_superclass] = empty_dict()

        accuracy_class_dict[pred_class][Tot] += 1
        accuracy_superclass_dict[pred_superclass][Tot] += 1
        accuracy_class_dict[pred_class][OCL] += len(other_classes.split(";"))
        accuracy_superclass_dict[pred_superclass][OCL] += len(other_superclasses.split(";"))

        for oc in other_classes.split(";"):
            if oc not in accuracy_class_dict:
                accuracy_class_dict[oc] = empty_dict()
            accuracy_class_dict[oc][Tot] += 1

        for oc in other_superclasses.split(";"):
            if oc not in accuracy_superclass_dict:
                accuracy_superclass_dict[oc] = empty_dict()
            accuracy_superclass_dict[oc][Tot] += 1

        accuracy_class_dict[pred_class][TF] += 1
        accuracy_superclass_dict[pred_superclass][TF] += 1

        if eval(correct):
            accuracy_class_dict[pred_class][Crr] += 1
            accuracy_superclass_dict[pred_superclass][Crr] += 1

            if distractor_class == target_class:
                accuracy_class_dict[pred_class][CTED] += 1
                accuracy_superclass_dict[pred_superclass][CTED] += 1
            else:
                accuracy_class_dict[pred_class][CTND] += 1
                accuracy_superclass_dict[pred_superclass][CTND] += 1
        else:
            if distractor_class == target_class:
                accuracy_class_dict[pred_class][WTED] += 1
                accuracy_superclass_dict[pred_superclass][WTED] += 1
            else:
                accuracy_class_dict[pred_class][WTND] += 1
                accuracy_superclass_dict[pred_superclass][WTND] += 1

    accuracy_class_dict = pd.DataFrame.from_dict(accuracy_class_dict)
    accuracy_superclass_dict = pd.DataFrame.from_dict(accuracy_superclass_dict)

    total = accuracy_superclass_dict.sum(axis=1)[Tot]

    accuracy_class_dict = normalize(accuracy_class_dict)
    accuracy_superclass_dict = normalize(accuracy_superclass_dict)
    return accuracy_class_dict, accuracy_superclass_dict, total


def coccurence(lines, classes, superclasses):
    cdf = pd.DataFrame(index=classes, columns=classes)
    scdf = pd.DataFrame(index=superclasses, columns=superclasses)
    cdf = cdf.fillna(0)
    scdf = scdf.fillna(0)

    for l in lines:
        data = split_line(l)
        target_class = data['target_class']
        distractor_class = data['distractor_class']

        target_superclass = data['target_superclass']
        distractor_superclass = data['distractor_superclass']

        cdf[target_class][distractor_class] += 1
        scdf[target_superclass][distractor_superclass] += 1

    return cdf, scdf


def analysis_df(infos, total):
    analysis = {}

    to_add = infos.loc[Acc, :].sum() / infos.shape[1]
    analysis['accuracy'] = to_add
    analysis[NObj] = int(total)

    return analysis


def accuracy_analysis(interaction_path, out_dir):
    console.log("Computing accuracy analysis...")

    with open(interaction_path, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)

    header = lines.pop(0)
    assert len(lines) > 0, "Empty Csv File!"
    class_infos, superclass_infos, total = get_infos(lines)

    cooc, scooc = coccurence(lines, class_infos.columns, superclass_infos.columns)
    class_analysis = analysis_df(class_infos, total)
    superclass_analysis = analysis_df(superclass_infos, total)

    cooc_path = out_dir.joinpath("acc_class_cooc.csv")
    scooc_path = out_dir.joinpath("acc_superclass_cooc.csv")
    class_infos_path = out_dir.joinpath("acc_class_infos.csv")
    superclass_infos_path = out_dir.joinpath("acc_superclass_infos.csv")
    class_analysis_path = out_dir.joinpath("acc_class_analysis.json")
    superclass_analysis_path = out_dir.joinpath("acc_superclass_analysis.json")

    cooc.to_csv(cooc_path)
    scooc.to_csv(scooc_path)
    class_infos.to_csv(class_infos_path)
    superclass_infos.to_csv(superclass_infos_path)

    with open(class_analysis_path, "w") as f:
        json.dump(class_analysis, f)

    with open(superclass_analysis_path, "w") as f:
        json.dump(superclass_analysis, f)

    print(f"Out saved in {out_dir}")

    return dict(
        acc_class_cooc=cooc,
        acc_superclass_cooc=scooc,
        acc_class_infos=class_infos,
        acc_superclass_infos=superclass_infos,
        acc_class_analysis=class_analysis,
        acc_superclass_analysis=superclass_analysis,
    )


if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    accuracy_analysis(interaction_path, out_dir)
