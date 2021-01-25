import json
import os
from glob import glob

import abbreviate
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

UUID_LEN = 8


def get_accumulator(path2accumulator, tag, mean_elem=5):
    size_guidance = {  # see below regarding this argument
        event_accumulator.COMPRESSED_HISTOGRAMS: 0,
        event_accumulator.IMAGES: 0,
        event_accumulator.AUDIO: 0,
        event_accumulator.SCALARS: 99999,
        event_accumulator.HISTOGRAMS: 0, }

    ea = event_accumulator.EventAccumulator(path2accumulator, size_guidance)
    ea = ea.Reload()
    try:
        ea = ea.Scalars(tag)[-mean_elem:]
    except KeyError:
        return None
    ea = [x.value for x in ea]
    ea = sum(ea) / mean_elem
    return ea


def get_params(path2params):
    path2params = glob(path2params + "/params*")[0]
    with open(path2params, "r") as f:
        params = json.load(f)
    return params


def filter_params(res_dict):
    all_params = [list(x.items()) for x in res_dict.values()]
    diffs = []
    for idx in range(len(all_params)):
        for jdx in range(idx, len(all_params)):
            diff = set(all_params[idx]) - set(all_params[jdx])
            diffs += [x[0] for x in diff]

    diffs = set(diffs)
    to_remove = set(
        ['num_workers', "tensorboard_dir", "random_seed", "resume_training", "checkpoint_dir", "log_dir", "n_epochs",
         "checkpoint_freq", "use_rich_traceback", "log_dir_uid", "train_log_prob", "test_log_prob", "test_logging_step",
         "train_logging_step", "data_root"])
    diffs -= to_remove

    for k, v in res_dict.items():
        res_dict[k] = {k2: v2 for k2, v2 in v.items() if k2 in diffs}

    if len(res_dict) == 0:
        return res_dict

    all_keys = [list(x.keys()) for x in list(res_dict.values())]
    m = np.argmin([len(x) for x in all_keys])
    filtered_keys = all_keys[m]

    for k, v in res_dict.items():
        res_dict[k] = {k2: v2 for k2, v2 in v.items() if k2 in filtered_keys}

    return res_dict


def find_logs(path2logs):
    logs = list(glob(path2logs + "/*"))
    res = {}
    for idx in range(len(logs)):
        log = logs[idx]
        if "random_signal" in log or "best" in log:
            continue
        params = get_params(log)
        uid = params['log_dir_uid'].split("/")[-1]

        res[uid] = params

    return res


def rename_logs(path, res_dict):
    if len(res_dict) == 0:
        return

    abbr = abbreviate.Abbreviate()
    for k, v in res_dict.items():
        v = {abbr.abbreviate(k1, target_len=5): v1 for k1, v1 in v.items()}
        new_name = str(v).replace("'", "") + "_" + k
        new_name = os.path.join(path, new_name)
        old_name = list(glob(path + f"/*{k}"))[0]
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            continue


def check_uid_name(log_path: str):
    logs = list(glob(log_path + "/*"))
    is_processed = False
    for lg in logs:
        lg = lg.split("/")[-1]

        if len(lg) == UUID_LEN:
            return 1
        elif "}" in lg:
            is_processed = True
    if is_processed:
        return -1
    else:
        return 0


def change_pipeline(log_path):
    check = check_uid_name(log_path)

    if check > 0:
        res = find_logs(log_path)
        res = filter_params(res)
        rename_logs(log_path, res)
    elif check == 0:
        logs = list(glob(log_path + "/*"))
        for lg in logs:
            change_pipeline(lg)
    else:
        return


if __name__ == '__main__':
    main_path = "./Logs"
    change_pipeline(main_path)
