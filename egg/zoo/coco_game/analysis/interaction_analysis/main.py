import json
import pickle
from pathlib import Path

import pandas as pd
from rich.progress import track

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.accuracy import accuracy_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.language import (
    ambiguity_richness,
    language_analysis,
)
from egg.zoo.coco_game.analysis.interaction_analysis.plotting import (
    plot_confusion_matrix,
    plot_multi_scatter,
)
from egg.zoo.coco_game.analysis.interaction_analysis.utils import add_row, path_parser


def load_generate_files(out_dir):
    files = list(out_dir.rglob("*.json"))
    files += list(out_dir.rglob("*.csv"))
    files += list(out_dir.rglob("*.pkl"))
    res_dict = {}

    for path in files:

        if path.suffix == ".csv":
            df = pd.read_csv(path, index_col=0)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                df = json.load(f)

        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                df = pickle.load(f)
        else:
            raise KeyError(f"Unrecognized stem {path.stem}")
        res_dict[path.stem] = df

    return res_dict


def get_analysis(interaction_path, out_dir):
    res_dict = load_generate_files(out_dir)

    if len(res_dict) == 0:
        acc_res = accuracy_analysis(interaction_path, out_dir)
        lan_res = language_analysis(interaction_path, out_dir)

        acc_res.update(lan_res)
        res_dict = acc_res

    return res_dict


class Analysis:
    def __init__(self, interaction_path: Path, out_dir: Path):
        analysis = get_analysis(interaction_path, out_dir)

        self.lang_sequence = analysis["lang_sequence"]
        self.lang_symbols = analysis["lang_symbol"]
        self.lang_sequence_cooc = analysis["lang_sequence_cooc"]
        self.lang_tensor = analysis["lang_tensor"]

        self.acc_class_infos = analysis["acc_class_infos"]
        self.acc_analysis = analysis["acc_analysis"]
        self.acc_class_cooc = analysis["acc_class_cooc"]

        self.cm_path = out_dir.joinpath("ConfusionMatrix")
        self.correlations_path = out_dir.joinpath("Correlations")
        self.language_tensor_path = out_dir.joinpath("LanguageTensor")

        self.cm_path.mkdir(parents=True, exist_ok=True)
        self.correlations_path.mkdir(parents=True, exist_ok=True)
        self.language_tensor_path.mkdir(parents=True, exist_ok=True)

        self.add_readme()

    def add_readme(self):

        readme_path = self.correlations_path.joinpath("README.md")
        explenations = [f"- *{k}* : {v}\n\n" for k, v in EXPLENATIONS.items()]

        with open(readme_path, "w+") as f:
            f.write("This file contains the correlation indices between any pair of implemented metric\n"
                    "Each figure reports a number of plots sorted by the corrleation value\n"
                    "Please find below the meaning of each metric:\n")

            f.writelines(explenations)

    def update_analysis(self):
        to_add = self.acc_class_infos.loc[ARt, :].corr(
            self.lang_symbols[CR]
        )
        self.acc_analysis = add_row(
            to_add, f"Correlation {ARt}-{SyCR}", self.acc_analysis
        )

        to_add = self.acc_class_infos.loc[Frq, :].corr(
            self.lang_symbols[CR]
        )
        self.acc_analysis = add_row(
            to_add, f"Correlation {Frq}-{SyCR}", self.acc_analysis
        )

        to_add = self.acc_class_infos.loc[ARt, :].corr(
            self.lang_sequence[CR]
        )
        self.acc_analysis = add_row(
            to_add, f"Correlation {ARt}-{SeCR}", self.acc_analysis
        )

        to_add = self.acc_class_infos.loc[Frq, :].corr(
            self.lang_sequence[CR]
        )
        self.acc_analysis = add_row(
            to_add, f"Correlation {Frq}-{SeCR}", self.acc_analysis
        )

    def update_infos(self):
        to_add, to_add2 = ambiguity_richness(self.lang_sequence_cooc)
        self.acc_class_infos = add_row(
            to_add, ARc, self.acc_class_infos
        )

        self.acc_class_infos = add_row(
            to_add2, f"{ARc}_perc", self.acc_class_infos
        )

    def plot_cm(self):
        plot_confusion_matrix(
            self.acc_class_cooc, "Class CoOccurence", save_dir=self.cm_path
        )

        to_plot = self.lang_symbols.drop(Frq)
        to_plot = to_plot.drop(CR, axis=1)
        plot_confusion_matrix(
            to_plot, "Class-Symbol CoOccurence", save_dir=self.cm_path
        )

        to_plot = self.lang_sequence.drop(Frq)
        to_plot = to_plot.drop(CR, axis=1)
        plot_confusion_matrix(
            to_plot, "Class-Sequence CoOccurence", save_dir=self.cm_path
        )

    def plot_correlations(self):

        info_len = len(self.acc_class_infos)
        for idx in track(range(info_len), "Plotting Correlations..."):
            metric = self.acc_class_infos.index[idx]
            to_plot = []

            for jdx in range(info_len):
                row_i = self.acc_class_infos.iloc[idx]
                row_j = self.acc_class_infos.iloc[jdx]

                # remove outlayers
                quant_i = row_i.between(row_i.quantile(0.05), row_i.quantile(0.95))
                quant_j = row_j.between(row_j.quantile(0.05), row_j.quantile(0.95))

                idexes = quant_i & quant_j

                row_i = row_i[idexes]  # without outliers
                row_j = row_j[idexes]  # without outliers

                # get correlation
                corr = row_i.corr(row_j)

                # cast tu numpy
                row_i = row_i.values
                row_j = row_j.values

                name_i = self.acc_class_infos.index[idx]
                name_j = self.acc_class_infos.index[jdx]

                to_plot.append((name_i, name_j, row_i, row_j, corr))

            to_plot = sorted(to_plot, key=lambda tup: tup[-1], reverse=True)
            path = self.correlations_path.joinpath(metric)
            path.mkdir(exist_ok=True)
            plot_multi_scatter(to_plot, save_dir=path, show=False)

    def plot_language_tensor(self):
        for k, df in track(self.lang_tensor.items(), "Plotting language tensor..."):

            if len(df) > 0:
                plot_confusion_matrix(
                    df, k, self.language_tensor_path, use_scaler=False, show=False
                )


if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    analysis = Analysis(interaction_path, out_dir)
    analysis.update_analysis()
    analysis.update_infos()
    analysis.plot_cm()
    analysis.plot_correlations()
    analysis.plot_language_tensor()
    a = 1
