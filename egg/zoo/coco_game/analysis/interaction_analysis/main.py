import json

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis.accuracy import accuracy_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.language import language_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.plotting import (
    plot_confusion_matrix,
)
from egg.zoo.coco_game.analysis.interaction_analysis.utils import add_row, path_parser


def load_generate_files(out_dir):
    files = list(out_dir.rglob("*"))
    res_dict = {}

    for path in files:

        if path.suffix == ".csv":
            df = pd.read_csv(path, index_col=0)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                df = json.load(f)

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
    def __init__(self, interaction_path, out_dir):
        analysis = get_analysis(interaction_path, out_dir)

        self.acc_class_cooc = analysis["acc_class_cooc"]
        self.lang_sequence = analysis["lang_sequence"]
        self.lang_symbols = analysis["lang_symbol"]
        self.lang_sequence_cooc = analysis["lang_sequence_cooc"]
        self.acc_class_infos = analysis["acc_class_infos"]
        self.acc_analysis = analysis["acc_analysis"]

    def update(self):
        to_add = self.acc_class_infos.loc["ambiguity_rate", :].corr(
            self.lang_symbols["class_richness"]
        )
        self.acc_analysis = add_row(
            to_add, "Correlation AmbiguityRate-SymbolClassRichness", self.acc_analysis
        )

        to_add = self.acc_class_infos.loc["frequency", :].corr(
            self.lang_symbols["class_richness"]
        )
        self.acc_analysis = add_row(
            to_add, "Correlation Frequency-SymbolClassRichness", self.acc_analysis
        )

        to_add = self.acc_class_infos.loc["ambiguity_rate", :].corr(
            self.lang_sequence["class_richness"]
        )
        self.acc_analysis = add_row(
            to_add, "Correlation AmbiguityRate-SeqClassRichness", self.acc_analysis
        )

        to_add = self.acc_class_infos.loc["frequency", :].corr(
            self.lang_sequence["class_richness"]
        )
        self.acc_analysis = add_row(
            to_add, "Correlation Frequency-SeqClassRichness", self.acc_analysis
        )

    def plot_cm(self):
        plot_confusion_matrix(self.acc_class_cooc, "Class CoOccurence")

        to_plot = self.lang_symbols.drop("frequency")
        to_plot = to_plot.drop("class_richness", axis=1)
        plot_confusion_matrix(to_plot, "Class-Symbol CoOccurence")

        to_plot = self.lang_sequence.drop("frequency")
        to_plot = to_plot.drop("class_richness", axis=1)
        plot_confusion_matrix(to_plot, "Class-Sequence CoOccurence")


if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    analysis = Analysis(interaction_path, out_dir)
    analysis.update()
    # analysis.plot_cm()
    a = 1
