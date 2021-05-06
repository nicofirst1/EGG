import pickle
from copy import copy
from pathlib import Path

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis import *
from egg.zoo.coco_game.analysis.interaction_analysis.joined import joined_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.utils import load_generate_files, max_sequence_num, normalize_drop, \
    add_row, console


class JoinedAnalysis:

    def __init__(self, interaction_path, out_dir, analysis_path: Path, class_analysis, superclass_analysis):
        self.class_analysis = class_analysis
        self.superclass_analysis = superclass_analysis
        self.interaction_path = interaction_path
        self.path_out_dir = out_dir

        model_idx = analysis_path.parts.index('runs') - 1
        self.model_name = analysis_path.parts[model_idx]
        console.log(f"Computing joined analysis for model {self.model_name}\n")

        filter = "joined_"
        res_dict = load_generate_files(analysis_path, filter)

        if not any(['joined' in x for x in res_dict.keys()]):
            joined_res = joined_analysis(interaction_path, analysis_path)
            res_dict.update(joined_res)

        for key in res_dict:
            new_key = key.replace(f"{filter}", "")
            self.__setattr__(new_key, res_dict[key])

        if "data" not in res_dict:
            self.data = {}
            self.same_superclass_sequence()
            self.super_class_comparison()

            path = analysis_path.joinpath(f"{filter}data.pkl")

            with open(path, "wb") as file:
                pickle.dump(self.data, file)

        self.add_readme()

    def readme_data_analysis(self, file):
        class_len = [len(x) - 2 for x in self.class_hierarchy.values()]
        accuracy = self.class_analysis.acc_analysis[Acc]

        file.write("# Data Analysis\n")
        file.write(
            f"In the following we perform an analysis for the model in '{self.interaction_path}' with an accuracy of {accuracy * 100:.2f}%\n"
            f"There are {len(self.class_hierarchy)} superclasses and {sum(class_len)} classes.\n"
            f"Each superclass has an average of {sum(class_len) / len(class_len):.3f} classes each.\n\n")

        file.write("## Class statistics\n"
                   "For each superclass, we report its weight in the dataset together with the weight of its class members:\n\n")

        sc_cooc = self.superclass_analysis.acc_infos
        c_cooc = self.class_analysis.acc_infos

        for superclass, dict in sorted(self.class_hierarchy.items()):

            dt = copy(dict)
            sc_total = dt.pop('total')
            sc_perc = dt.pop('total_perc')

            file.write(
                f"- **{superclass}** has {sc_total} instances which make up {sc_perc * 100:.2f}% of the dataset. It has an ambiguity rate of {sc_cooc[superclass][ARt] * 100:.2f}% and a intra class specificity of {sc_cooc[superclass][ISeS] * 100:.2f}%."
                f" Its classes are:\n")
            for class_name, perc in dt.items():
                file.write(
                    f"\t- *{class_name}* represent the {perc * 100:.2f}% of its superclass and has an ambiguity rate of {c_cooc[class_name][ARt] * 100:.2f}% and a intra class specificity of {sc_cooc[superclass][ISeS] * 100:.2f}%\n")
            file.write("\n")

        file.write("\n## General knowledge\n")

        to_drop_rows = [CTED, CTND, WTED, WTND, TF, POC, PSC, ARcP]

        file.write("\n### SuperClass\n")
        sc_cooc_min = sc_cooc.idxmin(axis=1)
        sc_cooc_max = sc_cooc.idxmax(axis=1)
        sc_cooc_min = sc_cooc_min.drop(to_drop_rows)
        sc_cooc_max = sc_cooc_max.drop(to_drop_rows)

        file.write("Following, some stats regarding the most peculiar values per superclass:\n")

        for row in sorted(sc_cooc_min.index):
            col = sc_cooc_min[row]
            val = sc_cooc[col][row]
            mean = sc_cooc.loc[row, :].sum() / sc_cooc.shape[1]
            val *= 100
            mean *= 100
            diff = abs(val - mean)
            file.write(
                f"- *{col}* has the minimum {row}; its value is {val:.2f}% which is {diff:.2f}% less than the mean value.\n")
        file.write("\n")

        for row in sc_cooc_max.index:
            col = sc_cooc_max[row]
            val = sc_cooc[col][row]
            mean = sc_cooc.loc[row, :].sum() / sc_cooc.shape[1]
            val *= 100
            mean *= 100
            diff = abs(val - mean)
            file.write(
                f"- *{col}* has the maximum {row}; its value is {val:.2f}% which is {diff:.2f}% more than the mean value.\n")

        file.write("\n### Class\n")
        c_cooc_min = c_cooc.idxmin(axis=1)
        c_cooc_max = c_cooc.idxmax(axis=1)

        c_cooc_min = c_cooc_min.drop(to_drop_rows)
        c_cooc_max = c_cooc_max.drop(to_drop_rows)

        file.write("Following, some stats regarding the most peculiar values per class:\n")

        for row in c_cooc_min.index:
            col = c_cooc_min[row]
            val = c_cooc[col][row]
            mean = c_cooc.loc[row, :].sum() / c_cooc.shape[1]
            val *= 100
            mean *= 100
            diff = abs(val - mean)
            file.write(
                f"- *{col}* has the minimum {row}; its value is {val:.2f}% which is {diff:.2f}% less than the mean value.\n")
        file.write("\n")
        for row in c_cooc_max.index:
            col = c_cooc_max[row]
            val = c_cooc[col][row]
            mean = c_cooc.loc[row, :].sum() / c_cooc.shape[1]
            val *= 100
            mean *= 100
            diff = abs(val - mean)
            file.write(
                f"- *{col}* has the maximum {row}; its value is {val:.2f}% which is {diff:.2f}% more than the mean value.\n")

    def readme_correlations(self, file):

        def get_special_corr(df, not_range):
            """
            Get all the values not between the not_range
            """
            not_range = [df[x].between(not_range[0], not_range[1]) for x in df.columns]
            not_range = pd.DataFrame(not_range)
            not_range = ~ not_range
            df = df[not_range]
            df = df[df < 0.99]
            return df

        def filter_same_words(main_word, serie):
            words = list(serie.index)
            words = [x.split("_") for x in words]
            keep = [all([mw not in words[i] for mw in main_word.split("_")]) for i in range(len(words))]
            return serie[keep].dropna()

        def write_correlations(file, cors, pavals):

            cors = get_special_corr(cors, [-0.4, 0.4])

            for col in cors.columns:
                c = cors[col]
                pval= pavals[col]

                c = filter_same_words(col, c)
                pval = filter_same_words(col, pval)

                if not (pval < 0.05).any():
                    continue

                pos = c[c > 0]
                neg = c[c < 0]

                pos = pos.sort_values(ascending=False)
                neg = neg.sort_values(ascending=False)

                file.write(f"- *{col}* :\n")

                for r in pos.index:
                    if pval[r] >0.05:
                        continue
                    file.write(f"\t **{r}** = {pos[r]:.3f} (pval {pval[r]:.3f})\n")
                for r in neg.index:
                    if pval[r] > 0.05:
                        continue
                    file.write(f"\t **{r}** = {neg[r]:.3f} (pval {pval[r]:.3f})\n")

                file.write("\n")

        file.write("\n# Correlations \n\n"
                   "In this section we report some correlations between metrics which stands out particularly\n")

        file.write("\n## Class corr\n")
        write_correlations(file, self.class_analysis.correlations, self.class_analysis.pvalues)
        file.write("\n## Superclass corr\n")
        write_correlations(file, self.superclass_analysis.correlations, self.superclass_analysis.pvalues)

    def readme_language_analysis(self, file):
        file.write("\n\n# Language Analysis\n")

        symbols = self.class_analysis.lang_symbol.columns[:-1]
        symbols = list(symbols)
        if '0' in symbols:
            symbols.remove('0')
        symbols = sorted(symbols)
        symbols_len = len(symbols)

        sequences = self.class_analysis.lang_sequence.columns[:-1]
        sequences = list(sequences)
        sequences_len = len(sequences)

        max_length = [len(x) for x in sequences]
        average_length = sum(max_length) / len(max_length)
        min_length = min(max_length)
        max_length = max(max_length)
        possible_sequences = max_sequence_num(symbols_len, max_length)

        file.write(
            f"The experiment considered a language with vocabulary size = {symbols_len} and max_length = {max_length}.\n"
            f"There are {sequences_len} unique sequences out of {possible_sequences} possibilities ({sequences_len / possible_sequences * 100:.2f}%).\n"
            f"The messages average length is {average_length:.2f}, with a minimum of {min_length}.\n")

        def symbols_analysis():
            file.write("\n## Symbols\n")
            file.write(
                f"There are {symbols_len} symbols as previously mentioned, for each we report its frequency, "
                f"distance from the random value (1/symbols = {1 / symbols_len:.2f}), most used superclass/class in a weighted context:\n\n")

            class_symbols_df = self.class_analysis.lang_symbol.copy()
            class_symbols_df = normalize_drop(class_symbols_df)

            superclass_symbols_df = self.superclass_analysis.lang_symbol.copy()
            superclass_symbols_df = normalize_drop(superclass_symbols_df)

            for sy in symbols:
                freq = self.superclass_analysis.lang_symbol[sy][Frq] * 100
                diff = freq / (1 / symbols_len * 100)

                most_used_class = class_symbols_df[sy].nlargest(2).index.tolist()
                most_used_superclass = superclass_symbols_df[sy].nlargest(2).index.tolist()

                class_diff12 = class_symbols_df[sy][most_used_class[0]] - class_symbols_df[sy][most_used_class[1]]
                superclass_diff12 = superclass_symbols_df[sy][most_used_superclass[0]] - superclass_symbols_df[sy][
                    most_used_superclass[1]]

                class_diff12 *= 100
                superclass_diff12 *= 100

                file.write(
                    f"- **{sy}** : has a frequency of {freq:.2f}%, which is {diff:.2f} times the random value.\n"
                    f"It is most used in *{most_used_superclass[0]}* (+{superclass_diff12:.2f}% than 2nd)-> *{most_used_class[0]}* (+{class_diff12:.2f}% than 2nd).\n\n")

        def sequences_analysis():
            file.write("\n## Sequences\n")
            n = 5
            sc_lang = self.superclass_analysis.lang_sequence.copy()
            frq_mean = sc_lang.loc[Frq].sum() / sc_lang.shape[1]
            frq_mean *= 100
            top_n_seqs = sc_lang.loc[Frq].nlargest(n)

            sc_lang = normalize_drop(sc_lang)

            c_lang = self.class_analysis.lang_sequence.copy()
            c_lang = normalize_drop(c_lang)

            file.write(
                f"The mean frequency per sequence is {frq_mean:.2f}% vs the random one  1/sequences= {1 / sequences_len * 100:.2f}%.\n")

            file.write("\n ## Top N\n"
                       f"The top {n} sequences by frequency are:\n")

            for idx in range(n):
                val = top_n_seqs[idx]
                sq = top_n_seqs.index[idx]
                val *= 100
                diff = val / frq_mean

                most_used_superclass = sc_lang[sq].nlargest(2).index.tolist()
                most_used_class = c_lang[sq].nlargest(2).index.tolist()

                class_diff12 = c_lang[sq][most_used_class[0]] - c_lang[sq][most_used_class[1]]
                superclass_diff12 = sc_lang[sq][most_used_superclass[0]] - sc_lang[sq][
                    most_used_superclass[1]]

                file.write(f"{idx + 1}. **{sq}** = {val:.2f}% which is {diff:.2f} times the mean value.\n"
                           f"It is most used in *{most_used_superclass[0]}* (+{superclass_diff12:.2f}% than 2nd)-> *{most_used_class[0]}* (+{class_diff12:.2f}% than 2nd).\n\n")

            file.write(f"\n ## Sequence Specificity\n")
            file.write(f"Another interesting aspect of the data is {SSeS}.\n"
                       f"{DEFINITIONS[SSeS]}\n"
                       f"Its value is {self.data[SSeS]:.3f}, which means that {self.data[SSeS] * 100:.1f}% of the {Se} "
                       f"({sequences_len * self.data[SSeS]:.2f}/{sequences_len}) is unique per superclass.\n"
                       f"This also means that {(1 - self.data[SSeS]) * 100:.1f}% is shared across the superclasses.\n"
                       f"\n")

            file.write(f"On the other hand, sequences may be unique on a class level, this is reported by the {CSeS}.\n"
                       f"{DEFINITIONS[CSeS]}\n"
                       f"Its value is {self.data[CSeS]:.3f}, which means that {self.data[CSeS] * 100:.1f}% of the {Se} "
                       f"({sequences_len * self.data[CSeS]:.2f}/{sequences_len}) is unique per class.\n"
                       f"This also means that {(1 - self.data[CSeS]) * 100:.1f}% is shared across the superclasses.\n"
                       f"\n")

            file.write(
                f"It is also important to consider how much of these sequences are shared across the members of a specific superclass."
                f" This is measured by *{SScS}*.\n"
                f"{DEFINITIONS[SScS]}\n"
                f"Its value is {self.data[SScS]:.4f}, which means that {self.data[SScS] * 100:.2f}% "
                f"of the *{SSeS}* is shared among classes of a specific superclass.\n"
                f"Indeed if we compute *{SScS}*X*{SSeS}*={self.data[SScS] * self.data[SSeS] * 100:.2f}% "
                f"we get the difference between *{SSeS}* and the *{CSeS}*.\n\n")

        symbols_analysis()
        sequences_analysis()

    def readme_classification_analysis(self, file):
        classPSC = self.class_analysis.acc_infos.loc[PSC].mean()
        classPOC = self.class_analysis.acc_infos.loc[POC].mean()

        superclassPSC = self.superclass_analysis.acc_infos.loc[PSC].mean()
        superclassPOC = self.superclass_analysis.acc_infos.loc[POC].mean()

        class_diff = abs(classPOC - classPSC)
        superclass_diff = abs(superclassPOC - superclassPSC)

        class_len = [len(x) for x in self.class_hierarchy.values()]

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

        corr_frq_serc = self.class_analysis.correlations[Frq][SeCR]
        pval_frq_serc = self.class_analysis.pvalues[Frq][SeCR]

        file.write(
            f"- There is a high correlation ({corr_frq_serc:.4f}) between the {Frq} and the {SeCR} (pval {pval_frq_serc:.4f}).\n"
            f"This implies that more frequent classes are mapped to more {Se}.\n\n")

        mpsc = self.data['general'].loc[PSC]
        mpoc = self.data['general'].loc[POC]
        file.write(
            f"- Given that there are {len(self.class_hierarchy)} superclasses and {class_len} classes; there is almost no difference in ({mpoc.loc['diff']:.3f}) when the target is different from the distractor between the superclass precision  ({mpoc.loc['superclass']:.3f}) and the class one  ({mpoc.loc['class']:.3f}).\n"
            f"On the other hand, the difference increases  ({mpsc.loc['diff']:.3f}) when the classes are the same; indeed it is easier when there are fewer cases such as in the superclass  ({mpsc.loc['superclass']:.3f}) than it is in the class instance  ({mpsc.loc['class']:.3f}).\n"
            f"")

    def add_readme(self):
        readme_path = self.path_out_dir.joinpath("README.md")

        with open(readme_path, "w+") as file:
            file.write("# Intro\n"
                       "Before starting the analysis, we report the meaning of the metrics used:\n")

            file.writelines(PRINT_DEF)
            file.write("\n\n")

            self.readme_data_analysis(file)

            self.readme_language_analysis(file)
            self.readme_correlations(file)

            self.readme_classification_analysis(file)

    def same_superclass_sequence(self, thrs=0.9999):
        seqclass = normalize_drop(self.superclass_analysis.lang_sequence)
        # normalize on column to sum to 1
        seqclass = seqclass.multiply(1 / seqclass.sum(), axis=1)
        tmp = {}
        total = 0
        # look for all the sequences that are mostly (> thrs) used for one superclass
        for superclass, classes in self.class_hierarchy.items():
            superclass_seqs = seqclass.loc[superclass, :] > thrs
            total += superclass_seqs.sum()
            tmp[superclass] = list(superclass_seqs[superclass_seqs].index)

        total /= seqclass.shape[1]
        self.data[SSeS] = total

        class_lang_sequence = normalize_drop(self.class_analysis.lang_sequence)
        # normalize on column to sum to 1
        class_lang_sequence = class_lang_sequence.multiply(1 / class_lang_sequence.sum(), axis=1)

        self.data[CSeS] = (class_lang_sequence > thrs).sum().sum() / class_lang_sequence.shape[1]


        total = 0
        idx = 0
        iseu_dict = {}
        # for alla the sequences above, look how much they are shared across classes of the same superclass
        for superclass, sequences in tmp.items():
            class_seq = class_lang_sequence[sequences]
            # filter classes
            class_seq = class_seq[(class_seq.T != 0).any()]
            non_zero = (class_seq.astype(bool).sum(axis=0) > 1).sum() / class_seq.shape[1]
            total += non_zero
            iseu_dict[superclass] = non_zero
            idx += 1

        total /= len(tmp)
        self.data[SScS] = total

        self.superclass_analysis.acc_infos = add_row(iseu_dict, SScS, self.superclass_analysis.acc_infos)
        self.superclass_analysis.compute_info_correlations()

    def super_class_comparison(self):
        """
        Use the super/class general dict to compare correlation and other metrics
        """

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
