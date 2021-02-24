import json

import pandas as pd

from egg.zoo.coco_game.analysis.interaction_analysis.accuracy import accuracy_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.language import language_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.utils import path_parser
import matplotlib.pyplot as plt

def load_generate_files(out_dir):
    files = list(out_dir.rglob("*"))
    res_dict = {}

    for path in files:

        if path.suffix == ".csv":
            df = pd.read_csv(path,index_col=0)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                df = json.load(f)

        else:
            raise KeyError(f"Unrecognized stem {path.stem}")
        res_dict[path.stem] = df

    return res_dict


def get_analysis(interaction_path, out_dir):
    res_dict=load_generate_files(out_dir)

    if len(res_dict)==0:
        acc_res=accuracy_analysis(interaction_path, out_dir)
        lan_res=language_analysis(interaction_path, out_dir)

        acc_res.update(lan_res)
        res_dict=acc_res

    return res_dict


if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    analysis=get_analysis(interaction_path, out_dir)
    acc_cooc = analysis["acc_cooc"]
    lang_sequence = analysis["lang_sequence"]
    lang_symbols = analysis["lang_symbol"]
    lang_sequence_cooc = analysis["lang_sequence_cooc"]
    acc_infos = analysis["acc_infos"]
    acc_analysis = analysis["acc_analysis"]

    corr_ambiguity_symbols_freq = acc_infos.loc["ambiguity_rate", :].corr(lang_symbols['class_richness'])
    acc_analysis["Correlation AmbiguityRate-SymbolFreq"] = corr_ambiguity_symbols_freq

    corr_ambiguity_sequence_freq = acc_infos.loc["ambiguity_rate", :].corr(lang_sequence['class_richness'])
    acc_analysis["Correlation AmbiguityRate-SequenceFreq"] = corr_ambiguity_sequence_freq

    plt.figure()
    acc_cooc.plot()
    a=1
    a=1
