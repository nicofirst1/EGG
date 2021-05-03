from pathlib import Path
from typing import List

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis import PRINT_DEF
from egg.zoo.coco_game.analysis.interaction_analysis.classes.JoinedAnalysis import JoinedAnalysis
from egg.zoo.coco_game.analysis.interaction_analysis.plotting import plot_multi_bar, plot_multi_bar4
from egg.zoo.coco_game.utils.utils import console


class ComparedAnalysis:

    def __init__(self, joined_list: List[JoinedAnalysis], out_dir, generate):
        self.joined_list = joined_list

        self.significance_thrs = 0.5
        self.plot = generate

        self.out_dir = Path(out_dir).joinpath("Comparison")
        self.readme_path = self.out_dir.joinpath("README.md")
        self.corr_plots_path = self.out_dir.joinpath("Correlations")
        self.corr_class_path = self.corr_plots_path.joinpath("Class")
        self.corr_superclass_path = self.corr_plots_path.joinpath("SuperClass")
        self.corr_both_path = self.corr_plots_path.joinpath("Both")

        console.log(f"Compared analysis for {len(joined_list)} models\n")

        self.out_dir.mkdir(exist_ok=True)
        self.corr_plots_path.mkdir(exist_ok=True)
        self.corr_class_path.mkdir(exist_ok=True)
        self.corr_superclass_path.mkdir(exist_ok=True)
        self.corr_both_path.mkdir(exist_ok=True)

        with open(self.corr_plots_path.joinpath("README.md"), "w+") as file:
            file.write("This folder contains a comaprison between the correlation factors of the models.\n"
                       "Each image contains the correlation between two metrics:\n"
                       "- A common one, which is reported in the title and the image name (e.g. accuracy.jpg)\n"
                       "- All other metrics labeled on the x axis\n\n"
                       "For each correlation pair both the values for two models are reported side by side (in different colors).\n"
                       "On top of the highes column the significance value is reported as a percentage. "
                       "The significance value report how much difference there is between the values for each models.\n")

        with open(self.readme_path, "w+") as file:
            file.write("# Intro\n"
                       "In the following we will analyze the differences between models in term of metrics.\n"
                       "The format for the analysis is always the same. "
                       "For all the metrics we report the values between model *i* and model *j* side by side.\n"
                       f"This value is reported for all the metrics with a significance level > {self.significance_thrs * 100}%.\n"
                       f"Before this, below you can find the definitions for all the metrics:\n")
            file.writelines(PRINT_DEF)

            self.iterate(file)

    def iterate(self, file):
        list_len = len(self.joined_list)
        for idx in range(list_len):
            join_i = self.joined_list[idx]

            for jdx in range(idx + 1, list_len):
                join_j = self.joined_list[jdx]

                file.write(f"\n# Difference {join_i.model_name}/{join_j.model_name}\n"
                           f"This section regards the differences between the model {join_i.model_name} located at `{join_i.interaction_path}`\n"
                           f"and the model {join_j.model_name}, located at `{join_j.interaction_path}`.\n")

                self.data_diff(file, join_i, join_j)
                self.class_diff(file, join_i, join_j, self.corr_class_path, superclass=False)
                self.class_diff(file, join_i, join_j, self.corr_superclass_path, superclass=True)

                if self.plot:
                    # plot correlation

                    corr_class = (join_i.class_analysis.correlations, join_j.class_analysis.correlations)
                    corr_superclass = (join_i.superclass_analysis.correlations, join_j.superclass_analysis.correlations)
                    plot_multi_bar4(corr_class, corr_superclass, (join_i.model_name, join_j.model_name),
                                    self.corr_both_path)

                    corr_class = (join_i.class_analysis.acc_analysis, join_j.class_analysis.acc_analysis)
                    corr_superclass = (join_i.superclass_analysis.acc_analysis, join_j.superclass_analysis.acc_analysis)


                    corr_class = [pd.DataFrame.from_dict(x, orient="index", columns=["General Info"]) for x in corr_class]
                    corr_superclass = [pd.DataFrame.from_dict(x, orient="index", columns=["General Info"]) for x in corr_superclass]
                    plot_multi_bar4(corr_class, corr_superclass, (join_i.model_name, join_j.model_name),
                                    self.out_dir)



    def data_diff(self, file, join_i, join_j):

        file.write(f"\n## Data difference\n")

        for k in join_i.data.keys():
            vi = join_i.data[k]
            vj = join_j.data[k]

            _ = write_diff(vi, vj, k, self.significance_thrs, file)

    def class_diff(self, file, join_i, join_j, path2plots, superclass=False):

        if superclass:
            file.write(f"\n## Superclass Difference\n")

            a_i = join_i.superclass_analysis
            a_j = join_j.superclass_analysis
        else:
            file.write(f"\n## Class Difference\n")

            a_i = join_i.class_analysis
            a_j = join_j.class_analysis

        file.write(f"\n### General Infos\n")
        _ = write_diff(a_i.acc_analysis, a_j.acc_analysis, "acc_analysis", self.significance_thrs, file)
        file.write(f"\n### Per class infos\n")
        _ = write_diff(a_i.acc_infos, a_j.acc_infos, "acc_infos", self.significance_thrs, file)
        file.write(f"\n### Correlations\n")
        intensity = write_diff(a_i.correlations, a_j.correlations, "correlations", self.significance_thrs, file)

        if self.plot:
            plot_multi_bar(a_i.correlations, a_j.correlations,
                           (join_i.model_name, join_j.model_name), intensity,
                           path2plots, superclass)


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

            file.write(f"- dataframe row **{row}** : \n")

            sig = sig.sort_values(ascending=False)[:max_cols]

            for col in sig.index:
                file.write(
                    f"\t- col *{col}* : {vi[col][row]:.3f}/{vj[col][row]:.3f}  ({sig[col] * 100:.2f}% significance)\n")
            file.write("\n")

    else:

        if significance > significance_thrs:
            file.write(f"- *{k}* : {vi:.3f}/{vj:.3f} ({significance * 100:.2f}% significance)\n")

    file.write("\n")
    return significance
