from pathlib import Path
from typing import List

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis import EXPLENATIONS
from egg.zoo.coco_game.analysis.interaction_analysis.classes.JoinedAnalysis import JoinedAnalysis


class ComparedAnalysis:

    def __init__(self, joined_list: List[JoinedAnalysis], out_dir):
        self.joined_list = joined_list

        self.significance_thrs = 0.5

        self.out_dir = Path(out_dir).joinpath("Comparison")
        self.readme_path = self.out_dir.joinpath("README.md")

        self.out_dir.mkdir(exist_ok=True)

        with open(self.readme_path, "w+") as file:
            file.write("# Intro\n"
                       "In the following we will analyze the differences between models in term of metrics.\n"
                       "The format for the analysis is always the same. "
                       "For all the metrics we report the values between model *i* and model *j* side by side.\n"
                       f"This value is reported for all the metrics with a significance level > {self.significance_thrs*100}%.\n"
                       f"Before this, below you can find the definitions for all the metrics:\n")
            explenations = [f"- *{k}* : {v}\n\n" for k, v in EXPLENATIONS.items()]
            file.writelines(explenations)

            self.iterate(file)

    def iterate(self, file):
        list_len = len(self.joined_list)
        for idx in range(list_len):
            join_i = self.joined_list[idx]

            for jdx in range(idx + 1, list_len):
                join_j = self.joined_list[jdx]

                file.write(f"\n# Difference {join_i.model_name}/{join_j.model_name}\n")

                self.data_diff(file, join_i, join_j)
                self.class_diff(file, join_i, join_j, superclass=False)
                self.class_diff(file, join_i, join_j, superclass=True)

    def data_diff(self, file, join_i, join_j):

        file.write(f"\n## Data difference\n")

        for k in join_i.data.keys():
            vi = join_i.data[k]
            vj = join_j.data[k]

            write_diff(vi, vj, k, self.significance_thrs, file)

    def class_diff(self, file, join_i, join_j, superclass=False):

        if superclass:
            file.write(f"\n## Superclass Difference\n")

            a_i = join_i.superclass_analysis
            a_j = join_j.superclass_analysis
        else:
            file.write(f"\n## Class Difference\n")

            a_i = join_i.class_analysis
            a_j = join_j.class_analysis

        file.write(f"\n### General Infos\n")
        write_diff(a_i.acc_analysis, a_j.acc_analysis, "acc_analysis", self.significance_thrs, file)
        file.write(f"\n### Per class infos\n")
        write_diff(a_i.acc_infos, a_j.acc_infos, "acc_infos", self.significance_thrs, file)
        file.write(f"\n### Correlations\n")
        write_diff(a_i.correlations, a_j.correlations, "correlations", self.significance_thrs, file)


def write_diff(vi, vj, k, significance_thrs, file, max_cols=5):
    assert type(vi) == type(vj), "Types must be the same"

    if isinstance(vi, dict):
        vi = pd.DataFrame.from_dict(vi, orient="index")
        vj = pd.DataFrame.from_dict(vj, orient="index")

    diff = abs(vi - vj)
    significance = abs(diff / abs(vi + vj))

    if isinstance(significance, pd.DataFrame):
        for row in significance.index:
            sig = significance.loc[row]

            # if significance value is not reached skip
            if not any(sig > significance_thrs):
                continue

            file.write(f"- dataframe row *{row}* : \n")

            sig = sig.sort_values(ascending=False)[:max_cols]

            for col in sig.index:
                file.write(
                    f"\t- col *{col}* : {vi[col][row]:.3f}/{vj[col][row]:.3f}  ({sig[col] * 100:.2f}% significance)\n")
            file.write("\n")

    else:

        if significance > significance_thrs:
            file.write(f"- *{k}* : {vi:.3f}/{vj:.3f} ({significance * 100:.2f}% significance)\n")

    file.write("\n")
