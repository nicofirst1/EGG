from pathlib import Path

import pandas as pd
import scipy
from rich.progress import track

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.accuracy import accuracy_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.language import language_analysis, ambiguity_richness
from egg.zoo.coco_game.analysis.interaction_analysis.plotting import plot_confusion_matrix, sort_dataframe, \
    plot_multi_scatter, plot_histogram
from egg.zoo.coco_game.analysis.interaction_analysis.utils import load_generate_files, console, add_row, \
    estimate_correlation, normalize_drop


def get_analysis(interaction_path, out_dir, filter):
    res_dict = load_generate_files(out_dir, filter)

    if not any(['acc' in x for x in res_dict.keys()]):
        acc_res = accuracy_analysis(interaction_path, out_dir)
        res_dict.update(acc_res)

    if not any(['lang' in x for x in res_dict.keys()]):
        lan_res = language_analysis(interaction_path, out_dir)
        res_dict.update(lan_res)

    return res_dict


class Analysis:
    def __init__(self, interaction_path: Path, out_dir: Path, analysis_path: Path, filter_str):

        analysis = get_analysis(interaction_path, analysis_path, filter_str)

        for key in analysis:
            new_key = key.replace(f"{filter_str}", "")
            self.__setattr__(new_key, analysis[key])

        self.path_out_dir = out_dir
        self.path_cm = out_dir.joinpath("ClassesOccurences")
        self.path_correlations = out_dir.joinpath("Correlations")
        self.path_lang_tensor = out_dir.joinpath("LanguageTensor")
        self.path_infos = out_dir.joinpath("ClassInfos")

        self.path_cm.mkdir(parents=True, exist_ok=True)
        self.path_correlations.mkdir(parents=True, exist_ok=True)
        self.path_lang_tensor.mkdir(parents=True, exist_ok=True)
        self.path_infos.mkdir(parents=True, exist_ok=True)

        console.log(
            f"Analyzing {filter_str} run with accuracy: {self.acc_analysis[Acc]:.3f} at path {interaction_path}."
            f"\nSaving results in {out_dir}")

        self.update_infos()
        self.update_analysis()
        self.compute_info_correlations()
        self.add_readme(interaction_path)

    def add_readme(self, interaction_path):

        readme_path = self.path_out_dir.joinpath("README.md")
        with open(readme_path, "w+") as f:
            f.write(
                f"This folder contain the output of the analysis ran on the interaction file `{interaction_path}` "
                f"which has an accuracy of  {self.acc_analysis[Acc]:.3f}\n"
                f"It is divided into subfolder each one reporting different informations about the interactions:\n"
                f"- *ClassesOccurences* : information based on the co-occurence of classes with other classes, symbols and sequences\n"
                f"- *ClassInfos* : a diverse set of metrics gatherd from the classes. Contains a csv and some histograms.\n"
                f"- *Correlations* : Has the previous metrics correlated with one another.\n"
                f"- *LanguageTensor* : information regarding the triple (target, distractor, sequence)\n"
                f"\nFind below the information regarding the metrics:\n"
            )

            f.writelines(PRINT_DEF)
            f.write("\n\n# Mean values\n"
                    "Next we report the mean values for some information about the analysis:\n")

            means = self.acc_infos.mean(axis=1)

            for ind in means.index:
                f.write(f"-Mean **{ind}** is : {means.loc[ind]:.4f}\n")

        readme_path = self.path_correlations.joinpath("README.md")

        with open(readme_path, "w+") as f:
            f.write("This folder contains the correlation indices between any pair of implemented metric\n"
                    "Each figure reports a number of plots sorted by the corrleation value\n"
                    "Please find below the meaning of each metric:\n")

            f.writelines(PRINT_DEF)

        readme_path = self.path_cm.joinpath("README.md")
        with open(readme_path, "w+") as f:
            f.write("This folder contains statistics about the co-occurence of a class with other metrics\n"
                    "Each point in the matrix is scaled in range 0-1 to appear more bright than it is, thus the figure may appear not normalized\n"
                    "Following the meaning of the plots:\n"
                    "- *ClassCoOccurence* : a cell ij reports the number a class i appears together with a class j \n"
                    "- *Class-Symbol CoOccurence*: a cell ij reports the number a symbol i appears together with a class j \n"
                    "- *Class-Sequence CoOccurence*: a cell ij reports the number a sequence i appears together with a class j \n"
                    "Moreover the two weighted figures are presented for the last two dataframes. The weightening is done based on the frequency of the class, i.e. the more frequent the less important.\n"
                    )

        readme_path = self.path_lang_tensor.joinpath("README.md")
        with open(readme_path, "w+") as f:
            f.write(
                "This folder contains an image for each class. The image name is the class when it is considered as a target\n"
                "The data shown inside is a matrix where the rows are a number of classes which appear as distractors with the current target class.\n"
                "The columns are the sequences appearing with that target.\n"
                "A cell in position ij (row-col) reports the times the sequence i is used with the distractor j, normalize by the total number of sequences for the target class.\n"
                "The image is sorted so to have the classes with the most sequences to the left\n"

            )

        readme_path = self.path_infos.joinpath("README.md")
        with open(readme_path, "w+") as f:
            f.write(
                "This folder encapsules the information coming from the lasses.\n"
                "You can find a file called *infos* which maps all the classes to various metrics (see below).\n"
                "Moreover each metric is plotted for all the classes (between the 95% and 5% percentile) in the *Histograms* folder\n\n"

            )
            f.writelines(PRINT_DEF)

    def update_analysis(self):

        self.acc_analysis[Sy] = self.lang_symbol.shape[1] - 1
        self.acc_analysis[Se] = self.lang_sequence.shape[1] - 1
        self.acc_analysis[Cls] = self.acc_infos.shape[1] - 1

        normalize_map = {
            TF: [Acc, CTND, CTED, WTED, WTND, ARt],
            Frq: [OCN, TF, Frq],
            Sy: [SyCR],
            Se: [SeCR],
            Cls: [ClsCmt],
            SeCR: [ARc, ARcP, ISeS]

        }
        nm = {}
        for k in normalize_map.keys():
            v = normalize_map[k]
            for k2 in v:
                nm[k2] = k

        # add means
        for row_id in self.acc_infos.index:
            row = self.acc_infos.loc[row_id].copy()

            if row_id in nm.keys():
                row_id2 = nm[row_id]

                try:
                    row2 = self.acc_infos.loc[row_id2]

                except KeyError:

                    row2 = self.acc_analysis[row_id2]
            else:
                row2 = 1

            row /= row2
            row = row.mean()

            self.acc_analysis[row_id] = row

    def update_infos(self):

        to_add = self.lang_sequence[CR]
        self.acc_infos = add_row(
            to_add, SeCR, self.acc_infos
        )

        to_add = self.lang_symbol[CR]
        self.acc_infos = add_row(
            to_add, SyCR, self.acc_infos
        )

        to_add, to_add2 = ambiguity_richness(self.lang_sequence_cooc)
        self.acc_infos = add_row(
            to_add, ARc, self.acc_infos
        )

        self.acc_infos = add_row(
            to_add2, ClsCmt, self.acc_infos
        )

        shared_appearances = {k: len(v) for k, v in self.lang_sequence_cooc.items()}
        self.acc_infos = add_row(
            shared_appearances, ARcP, self.acc_infos
        )

        lang_sequence = normalize_drop(self.lang_sequence, axis=0)

        # get all the sequences that are used once
        sequences = (lang_sequence == 1).any()
        sequences = sequences[sequences].index
        # count the unique sequence per class
        ises = (lang_sequence[sequences] == 1).sum(axis=1)

        # total = self.acc_infos.loc[Frq] * self.acc_analysis[NObj] * self.acc_infos.loc[TF]
        # ises /= total

        # # normalize by number of sequences
        # ises = ises / lang_sequence.shape[1]
        # # divide per target frequency
        # ises = ises * self.acc_infos.loc[TF]
        # add to acc infos
        self.acc_infos = add_row(
            ises, ISeS, self.acc_infos
        )

    def plot_cm(self):
        plot_confusion_matrix(
            self.acc_cooc, "Class CoOccurence", save_dir=self.path_cm, show=False
        )

        to_plot = self.lang_symbol.drop(Frq)
        to_plot = to_plot.drop(CR, axis=1)
        to_plot = sort_dataframe(to_plot, True)

        plot_confusion_matrix(
            to_plot, "Class-Symbol CoOccurence", save_dir=self.path_cm, show=False
        )

        to_plot = self.lang_sequence.drop(Frq)
        to_plot = to_plot.drop(CR, axis=1)
        to_plot = sort_dataframe(to_plot, True)
        plot_confusion_matrix(
            to_plot, "Class-Sequence CoOccurence", save_dir=self.path_cm, show=False
        )

        class_freq = self.acc_infos.loc[Frq]

        to_plot = self.lang_symbol.drop(Frq)
        to_plot = to_plot.drop(CR, axis=1)
        to_plot = to_plot.multiply(1 - class_freq, axis=0)
        to_plot = sort_dataframe(to_plot, True)

        plot_confusion_matrix(
            to_plot, "Class-Symbol CoOccurence Weighted", save_dir=self.path_cm, show=False
        )

        to_plot = self.lang_sequence.drop(Frq)
        to_plot = to_plot.drop(CR, axis=1)
        to_plot = to_plot.multiply(1 - class_freq, axis=0)
        to_plot = sort_dataframe(to_plot, True)

        plot_confusion_matrix(
            to_plot, "Class-Sequence CoOccurence Weighted", save_dir=self.path_cm, show=False
        )

    def compute_info_correlations(self):
        indices = list(self.acc_infos.index)
        df_corr = pd.DataFrame(columns=indices, index=indices, dtype=float)
        df_pval = pd.DataFrame(columns=indices, index=indices, dtype=float)

        for idx in indices:
            for jdx in indices:
                corr = estimate_correlation(self.acc_infos, idx, jdx)

                df_corr[idx][jdx] = corr[0]
                df_pval[idx][jdx] = corr[1]

        self.correlations = df_corr
        self.pvalues = df_pval

    def plot_correlations(self):
        info_len = len(self.acc_infos)
        for idx in track(range(info_len), "Plotting Correlations..."):
            metric = self.acc_infos.index[idx]
            to_plot = []

            for jdx in range(info_len):
                row_i = self.acc_infos.iloc[idx]
                row_j = self.acc_infos.iloc[jdx]

                # remove outlayers
                quant_i = row_i.between(row_i.quantile(0.05), row_i.quantile(0.95))
                quant_j = row_j.between(row_j.quantile(0.05), row_j.quantile(0.95))

                idexes = quant_i & quant_j

                row_i = row_i[idexes]  # without outliers
                row_j = row_j[idexes]  # without outliers

                # get correlation
                corr, pvalue = scipy.stats.pearsonr(row_i, row_j)

                # cast tu numpy
                row_i = row_i.values
                row_j = row_j.values

                name_i = self.acc_infos.index[idx]
                name_j = self.acc_infos.index[jdx]

                to_plot.append((name_i, name_j, row_i, row_j, corr, pvalue))

            to_plot = sorted(to_plot, key=lambda tup: tup[-1], reverse=True)
            path = self.path_correlations.joinpath(metric)
            path.mkdir(exist_ok=True)
            plot_multi_scatter(to_plot, save_dir=path, show=False)

    def plot_2d_language_tensor(self):
        for k, df in track(self.lang_tensor.items(), "Plotting language tensor..."):

            if len(df) > 0:
                df = sort_dataframe(df, False)

                plot_confusion_matrix(
                    df, k, self.path_lang_tensor, use_scaler=False, show=False
                )

    def plot_infos(self):

        csv_path = self.path_infos.joinpath("infos.csv")
        self.acc_infos.to_csv(csv_path)

        images_path = self.path_infos.joinpath("Histograms")
        images_path.mkdir(exist_ok=True)

        for idx in track(range(len(self.acc_infos)), "Plotting Class infos"):
            metric = self.acc_infos.index[idx]
            data = self.acc_infos.iloc[idx]
            quant = data.between(data.quantile(0.05), data.quantile(0.95))
            data = data[quant]

            plot_histogram(data, metric, images_path, show=False)
