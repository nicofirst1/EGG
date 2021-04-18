from pathlib import Path

from rich.progress import track

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.accuracy import accuracy_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.joined import joined_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.language import language_analysis, ambiguity_richness, \
    class_richness
from egg.zoo.coco_game.analysis.interaction_analysis.plotting import plot_confusion_matrix, sort_dataframe, \
    plot_multi_scatter, plot_histogram
from egg.zoo.coco_game.analysis.interaction_analysis.utils import load_generate_files, console, add_row


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

        self.add_readme(interaction_path)

        console.log(
            f"Analyzing {filter_str} run with accuracy: {self.acc_analysis.iloc[0][0]:.3f} at path {interaction_path}."
            f"\nSaving results in {out_dir}")

        self.update_analysis()
        self.update_infos()

    def add_readme(self, interaction_path):
        explenations = [f"- *{k}* : {v}\n\n" for k, v in EXPLENATIONS.items()]

        readme_path = self.path_out_dir.joinpath("README.md")
        with open(readme_path, "w+") as f:
            f.write(
                f"This folder contain the output of the analysis ran on the interaction file `{interaction_path}` "
                f"which has an accuracy of  {self.acc_analysis.iloc[0][0]:.3f}\n"
                f"It is divided into subfolder each one reporting different informations about the interactions:\n"
                f"- *ClassesOccurences* : information based on the co-occurence of classes with other classes, symbols and sequences\n"
                f"- *ClassInfos* : a diverse set of metrics gatherd from the classes. Contains a csv and some histograms.\n"
                f"- *Correlations* : Has the previous metrics correlated with one another.\n"
                f"- *LanguageTensor* : information regarding the triple (target, distractor, sequence)\n"
                f"\nFind below the information regarding the metrics:\n"
            )
            f.writelines(explenations)

        readme_path = self.path_correlations.joinpath("README.md")

        with open(readme_path, "w+") as f:
            f.write("This folder contains the correlation indices between any pair of implemented metric\n"
                    "Each figure reports a number of plots sorted by the corrleation value\n"
                    "Please find below the meaning of each metric:\n")

            f.writelines(explenations)

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
            f.writelines(explenations)

    def update_analysis(self):
        to_add = self.acc_infos.loc[ARt, :].corr(
            self.lang_symbol[CR]
        )
        self.acc_analysis = add_row(
            float(to_add), f"Corr {ARt}-{SyCR}", self.acc_analysis
        )

        to_add = self.acc_infos.loc[Frq, :].corr(
            self.lang_symbol[CR]
        )
        self.acc_analysis = add_row(
            to_add, f"Corr {Frq}-{SyCR}", self.acc_analysis
        )

        to_add = self.acc_infos.loc[ARt, :].corr(
            self.lang_sequence[CR]
        )
        self.acc_analysis = add_row(
            to_add, f"Corr {ARt}-{SeCR}", self.acc_analysis
        )

        to_add = self.acc_infos.loc[Frq, :].corr(
            self.lang_sequence[CR]
        )
        self.acc_analysis = add_row(
            to_add, f"Corr {Frq}-{SeCR}", self.acc_analysis
        )

        # add means
        for row in self.acc_infos.index:
            mean = self.acc_infos.loc[row].mean()
            self.acc_analysis = add_row(mean, f"mean {row}", self.acc_analysis)

    def update_infos(self):
        to_add, to_add2 = ambiguity_richness(self.lang_sequence_cooc)
        self.acc_infos = add_row(
            to_add, ARc, self.acc_infos
        )

        self.acc_infos = add_row(
            to_add2, f"{ARc}_perc", self.acc_infos
        )

        to_add = class_richness(self.lang_sequence_cooc)
        self.acc_infos = add_row(
            to_add, CR, self.acc_infos
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
                corr = row_i.corr(row_j)

                # cast tu numpy
                row_i = row_i.values
                row_j = row_j.values

                name_i = self.acc_infos.index[idx]
                name_j = self.acc_infos.index[jdx]

                to_plot.append((name_i, name_j, row_i, row_j, corr))

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

    def add_infos(self):

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


class JoinedAnalysis:

    def __init__(self, interaction_path, out_dir, analysis_path, class_analysis, superclass_analysis):
        self.class_analysis = class_analysis
        self.superclass_analysis = superclass_analysis

        filter = "joined_"
        res_dict = load_generate_files(analysis_path, filter)

        if not any(['joined' in x for x in res_dict.keys()]):
            joined_res = joined_analysis(interaction_path, out_dir)
            res_dict.update(joined_res)

        for key in res_dict:
            new_key = key.replace(f"{filter}", "")
            self.__setattr__(new_key, res_dict[key])

        self.path_out_dir = out_dir
        self.data = {}

    def add_readme(self):
        readme_path = self.path_out_dir.joinpath("README.md")

        classPSC = self.class_analysis.acc_infos.loc[PSC].mean()
        classPOC = self.class_analysis.acc_infos.loc[POC].mean()

        superclassPSC = self.superclass_analysis.acc_infos.loc[PSC].mean()
        superclassPOC = self.superclass_analysis.acc_infos.loc[POC].mean()

        class_diff = abs(classPOC - classPSC)
        superclass_diff = abs(superclassPOC - superclassPSC)

        with open(readme_path, "w+") as file:
            file.write(
                f"In the following file some metrics are reported together with a brief analysis\n"
                f"Considering the discrimination objective (predicting the correct target) as a classification one (predicting the class of the target) we have that\n\n"
                f"- The prediction precision when the target is the class as the distractor is {classPSC:.3f} vs when is not {classPOC:.3f}\n"
                f"The difference between the two is {class_diff:.3f}, which implies that it is easier to classify object when they belong to  "
            )

            if classPSC > classPOC:
                file.write(f"the same class")
            else:
                file.write("different classes")

            file.write(".\n\n"
                       f"- On the other hand the same kind of precision on the superclasses is {superclassPSC:.3f} when target==distractor and {superclassPOC:.3f} otherwise.\n"
                       f"Still the difference between the two implies that is easier to classify images belonging to different superclasses")

            if superclass_diff < class_diff:
                file.write(
                    f", although the difference ({superclass_diff:.3f}) is not as important as the one xclass one.\n")
            else:
                file.write(f"TODO")

            file.write("\n\n")
            sequence_len = self.class_analysis.lang_sequence.shape[1]
            file.write(f"Another interesting aspect of the data is {SeS}.\n"
                       f"{SeS} is defined as: {EXPLENATIONS[SeS]}\n"
                       f"Its value is {self.data[SeS]:.3f}, which means that {self.data[SeS] * 100:.1f}% of the {Se} ({sequence_len * self.data[SeS]:.2f}/{sequence_len}) is unique per superclass.\n\n"
                       f"It is also important to consider how much of these symbols are shared across the members of a specific superclass. This is measured by {ISeU}.\n"
                       f"The {ISeU} is defined as: {EXPLENATIONS[ISeU]}\n"
                       f"Its value is {self.data[ISeU]:3f}\n\n")

            corr_frq_serc = self.class_analysis.acc_analysis.loc[f"Corr {Frq}-{SeCR}"]
            corr_frq_serc = corr_frq_serc[0]
            file.write(
                f"- There is a high correlation ({corr_frq_serc:.4f}) between the {Frq} and the {SeCR} defined as: {EXPLENATIONS[SeCR]}\n"
                f"This implies that more frequent classes are mapped to more {Se}.\n")

        a = 1

    def column_normalization(self, df):
        df = df.drop('frequency')
        df = df.multiply(1 / df['class_richness'], axis=0)
        df = df.drop('class_richness', axis=1)
        df = df.multiply(1 / df.sum(), axis=1)
        return df

    def same_superclass_sequence(self):

        seqclass = self.column_normalization(self.superclass_analysis.lang_sequence)

        tmp = {}
        thrs = 0.99
        total = 0
        for superclass, classes in self.class_hierarchy.items():
            superclass_seqs = seqclass.loc[superclass, :] > thrs
            total += superclass_seqs.sum()
            tmp[superclass] = list(superclass_seqs[superclass_seqs].index)

        total /= seqclass.shape[1]
        self.data[SeS] = total

        lang_sequence = self.column_normalization(self.class_analysis.lang_sequence)

        total = 0
        idx = 0
        for superclass, sequences in tmp.items():
            class_seq = lang_sequence[sequences]
            class_seq = class_seq[(class_seq.T != 0).any()]
            non_zero = class_seq.astype(bool).sum(axis=0) / class_seq.shape[0] - 1 / class_seq.shape[0]
            non_zero = sum(non_zero) / len(non_zero)
            total += non_zero
            idx += 1

        total /= idx
        self.data[ISeU] = total

    def add_meningful_data(self):
        a = 1
