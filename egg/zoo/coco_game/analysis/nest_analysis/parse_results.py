import ast
import json

import numpy as np
import pandas as pd

# prov_lines = """
# {"loss": 0.6343242526054382, "f_loss": 0.3030605614185333, "x_loss": 0.7756949663162231, "kl_loss": 2.823378086090088, "accuracy_receiver": 0.5036764740943909, "accuracy_sender": 0.1875, "custom_loss": 0.7756949663162231, "sender_entropy": 0.15114206075668335, "receiver_entropy": 0.0, "length": 3.808823585510254, "policy_loss": -0.28811606764793396, "weighted_entropy": 0.015770524740219116, "mode": "train", "epoch": 1}
# {"loss": 0.7026403546333313, "f_loss": 0.18539749085903168, "x_loss": 0.7048177123069763, "kl_loss": 2.6629514694213867, "accuracy_receiver": 0.46875, "accuracy_sender": 0.3333333432674408, "custom_loss": 0.7048177123069763, "sender_entropy": 1.240893103332183e-13, "receiver_entropy": 0.0, "length": 4.0, "policy_loss": 0.0, "weighted_entropy": 1.240893103332183e-14, "mode": "test", "epoch": 1}
# {"loss": 0.6999519467353821, "f_loss": 0.18471960723400116, "x_loss": 0.7030501365661621, "kl_loss": 2.661822557449341, "accuracy_receiver": 0.4797794222831726, "accuracy_sender": 0.29963234066963196, "custom_loss": 0.7030501365661621, "sender_entropy": 1.6770028562432954e-13, "receiver_entropy": 0.0, "length": 4.0, "policy_loss": 0.0, "weighted_entropy": 1.677002924005931e-14, "mode": "train", "epoch": 2}
# {"loss": 0.6870132684707642, "f_loss": 0.17025837302207947, "x_loss": 0.6849746108055115, "kl_loss": 2.642674446105957, "accuracy_receiver": 0.56640625, "accuracy_sender": 0.3125, "custom_loss": 0.6849746108055115, "sender_entropy": 1.985521461329333e-14, "receiver_entropy": 0.0, "length": 4.0, "policy_loss": 0.0, "weighted_entropy": 1.985521461329333e-15, "mode": "test", "epoch": 2}
# """
# prov_lines = prov_lines.split("\n")
from egg.zoo.coco_game.analysis.nest_analysis.nest_utils import path_parser


def get_configs(lines: list) -> str:
    def parse_config_lines(lines):
        start = "parsers.py"
        stop = "]"
        skip_start = "("
        skip_end = "),"

        configs = []
        to_append = False
        for l in lines:
            l = l.strip()
            if start in l:
                to_append = True
            elif stop == l:
                break
            elif l == skip_start:
                to_append = False

            if to_append:
                configs.append(l)

            if l == skip_end:
                to_append = True

        configs.pop(0)

        configs = [x for x in configs if "device" not in x]

        return configs

    configs = parse_config_lines(lines)
    configs = "[" + "".join(configs) + "]"
    configs = ast.literal_eval(configs)
    return dict(configs)


def get_best_result(epochs, tag):
    # convert to dict
    epochs = [json.loads(x) for x in epochs]

    test = [x for x in epochs if x["mode"] == "test"]
    train = [x for x in epochs if x["mode"] == "train"]

    best_idx = -1
    best_tag = -1
    for idx in range(len(test)):

        tag_val = test[idx][tag]
        if tag_val > best_tag:
            best_tag = tag_val
            best_idx = idx

    best_vals = [train[best_idx], test[best_idx]]
    max_epochs = test[idx]["epoch"]

    return best_vals, max_epochs


def parse_results(nest_path, tag):
    files = list(nest_path.rglob("*.out"))

    res = []
    for out_file in files:
        with open(out_file, "r") as f:
            lines = f.readlines()

        if "submitit ERROR" in "\n".join(lines):
            continue

        configs = get_configs(lines)
        # lines = prov_lines
        epochs = [x for x in lines if tag in x]
        best_vals, max_epoch = get_best_result(epochs, tag)

        res.append(
            dict(
                configs=configs,
                best_vals=best_vals,
                max_epoch=max_epoch,
            )
        )
    return res


def build_dataframe(results):
    tmp = results[0]
    config_cols = sorted(list(tmp["configs"].keys()))
    train_cols = sorted([f"train_{x}" for x in tmp["best_vals"][0].keys()])
    test_col = sorted([f"test_{x}" for x in tmp["best_vals"][0].keys()])
    last_col = ["max_epoch"]
    columns = config_cols + train_cols + test_col + last_col

    df = pd.DataFrame(columns=columns)

    idx = 0
    for tmp in results:
        row = []
        for k in config_cols:
            row.append(tmp['configs'][k])
        for k in train_cols:
            k = k.replace("train_", "")
            row.append(tmp['best_vals'][0][k])
        for k in test_col:
            k = k.replace("test_", "")
            row.append(tmp['best_vals'][1][k])

        row += [tmp["max_epoch"]]
        df.loc[idx] = row
        idx += 1

    return df


def config_check(configs):
    all_lens = [len(x) for x in configs]
    # all configs are the same
    if len(set(all_lens)) == 1: return configs

    all_lens = np.array(all_lens)
    max_idx = np.argmax(all_lens)
    min_idx = np.argmin(all_lens)
    c_max = configs[max_idx]
    c_min = configs[min_idx]
    to_add_key = set(c_max.keys()) - set(c_min.keys())

    for idx in range(len(configs)):
        for k in to_add_key:
            if k not in configs[idx].keys():
                configs[idx][k] = ""
    return configs


if __name__ == "__main__":
    nest_path = path_parser()
    csv_name = nest_path.joinpath(f"results_{nest_path.stem}.csv")

    ids = parse_results(nest_path, tag="accuracy_receiver")
    configs = config_check([x['configs'] for x in ids])

    for idx in range(len(configs)):
        ids[idx]['configs'] = configs[idx]

    df = build_dataframe(ids)

    df.to_csv(csv_name)
    print(f"File save in {csv_name}")
