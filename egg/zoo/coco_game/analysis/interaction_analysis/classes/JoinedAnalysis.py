from copy import copy

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.joined import joined_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.utils import load_generate_files, max_sequence_num


class JoinedAnalysis:

    def __init__(self, interaction_path, out_dir, analysis_path, class_analysis, superclass_analysis):
        self.class_analysis = class_analysis
        self.superclass_analysis = superclass_analysis

        filter = "joined_"
        res_dict = load_generate_files(analysis_path, filter)

        if not any(['joined' in x for x in res_dict.keys()]):
            joined_res = joined_analysis(interaction_path, analysis_path)
            res_dict.update(joined_res)

        for key in res_dict:
            new_key = key.replace(f"{filter}", "")
            self.__setattr__(new_key, res_dict[key])

        self.interaction_path = interaction_path
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
            class_len = [len(x) for x in self.class_hierarchy.values()]
            accuracy = self.class_analysis.acc_analysis[Acc]

            file.write("# Data Analysis\n")
            file.write(
                f"In the following we perform an analysis for the model in '{self.interaction_path}' with an accuracy of {accuracy:.3f}%\n"
                f"There are {len(self.class_hierarchy)} superclass and {sum(class_len)} classes.\n"
                f"Each superclass has an average of {sum(class_len) / len(class_len):.3f} classes each.\n\n")

            file.write("## Class statistics\n"
                       "For each superclass, we report its weight in the dataset together with the weight of its class members:\n\n")

            sc_cooc = self.superclass_analysis.acc_cooc.copy()
            sc_cooc /= sc_cooc.sum()
            sc_cooc *= 100

            c_cooc = self.class_analysis.acc_cooc.copy()
            c_cooc /= c_cooc.sum()
            c_cooc *= 100

            for superclass, dict in self.class_hierarchy.items():

                dt = copy(dict)
                sc_total = dt.pop('total')
                sc_perc = dt.pop('total_perc')

                file.write(
                    f"- **{superclass}** has {sc_total} instances which make up {sc_perc * 100:.2f}% of the dataset. It has an ambiguity rate of {sc_cooc[superclass][superclass]:.2f}%."
                    f" Its classes are:\n")
                for class_name, perc in dt.items():
                    file.write(f"\t- *{class_name}* represent the {perc * 100:.2f}% of its superclass and has an ambiguity rate of {c_cooc[class_name][class_name]:.2f}%\n")
                file.write("\n")

            file.write("\n# Language Analysis\n")
            symbols = self.class_analysis.lang_symbol.shape[1]
            sequences = self.class_analysis.lang_sequence.shape[1]
            max_length = [len(x) for x in self.class_analysis.lang_sequence.columns[:-1]]
            average_length = sum(max_length) / len(max_length)
            min_length = min(max_length)
            max_length = max(max_length)
            possible_symbols = max_sequence_num(symbols, max_length)

            file.write(
                f"The experiment considered a language with vocabulary size = {symbols} and max_length = {max_length}.\n"
                f"There are {sequences} unique sequences out of {possible_symbols} possibilities ({sequences / possible_symbols * 100:.2f}%).\n"
                f"The messages average length is {average_length}, with a minimum of {min_length}.\n")

            file.write("\n\n# Classification \n")

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

            corr_frq_serc = self.class_analysis.acc_analysis[f"Corr {Frq}-{SeCR}"]

            file.write(
                f"- There is a high correlation ({corr_frq_serc:.4f}) between the {Frq} and the {SeCR} defined as: {EXPLENATIONS[SeCR]}\n"
                f"This implies that more frequent classes are mapped to more {Se}.\n\n")

            mpsc = self.data['general'].loc[PSC]
            mpoc = self.data['general'].loc[POC]
            file.write(
                f"- Given that there are {len(self.class_hierarchy)} superclasses and {class_len} classes; there is almost no difference in ({mpoc.loc['diff']:.3f}) when the target is different from the distractor between the superclass precision  ({mpoc.loc['superclass']:.3f}) and the class one  ({mpoc.loc['class']:.3f}).\n"
                f"On the other hand, the difference increases  ({mpsc.loc['diff']:.3f}) when the classes are the same; indeed it is easier when there are fewer cases such as in the superclass  ({mpsc.loc['superclass']:.3f}) than it is in the class instance  ({mpsc.loc['class']:.3f}).\n"
                f"")

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

        ca_anal = self.class_analysis.acc_analysis
        sca_anal = self.superclass_analysis.acc_analysis

        ca_anal = pd.DataFrame.from_dict(ca_anal, orient='index')
        sca_anal = pd.DataFrame.from_dict(sca_anal, orient='index')
        cat = pd.concat([ca_anal, sca_anal], axis=1)
        cat.columns = ["class", "superclass"]
        cat = cat[(cat <= 1).all(axis=1)]
        diff = abs(cat['class'] - cat['superclass'])
        cat['diff'] = diff

        self.data['general'] = cat
