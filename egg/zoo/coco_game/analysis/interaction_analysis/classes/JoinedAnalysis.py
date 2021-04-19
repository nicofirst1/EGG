import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.joined import joined_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.utils import load_generate_files


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
                f"This implies that more frequent classes are mapped to more {Se}.\n\n")

            class_len=sum([len(x) for x in self.class_hierarchy.values()])
            mpsc = self.data['general'].loc[PSC]
            mpoc = self.data['general'].loc[POC]
            file.write(f"- Given that there are {len(self.class_hierarchy)} superclasses and {class_len} classes; there is almost no difference in ({mpoc.loc['diff']:.3f}) when the target is different from the distractor between the superclass precision  ({mpoc.loc['superclass']:.3f}) and the class one  ({mpoc.loc['class']:.3f}).\n"
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
        cat = pd.concat([ca_anal, sca_anal], axis=1)
        cat.columns = ["class", "superclass"]
        cat = cat[(cat <= 1).all(axis=1)]
        diff = abs(cat['class'] - cat['superclass'])
        cat['diff'] = diff

        self.data['general'] = cat
        a = 1
