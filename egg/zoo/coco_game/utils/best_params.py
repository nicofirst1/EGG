import json
import os
from glob import glob

import abbreviate
import numpy as np
from rich.console import Console
from rich.progress import track
from tensorboard.backend.event_processing import event_accumulator


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


def find_accumulators(path2logs):
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
    abbr = abbreviate.Abbreviate()
    for k, v in res_dict.items():
        v={abbr.abbreviate(k1, target_len=5):v1 for k1,v1 in v.items()}
        new_name = str(v).replace("'", "") + "_" + k
        new_name = os.path.join(path, new_name)
        old_name = os.path.join(path, k)
        os.rename(old_name, new_name)


if __name__ == '__main__':
    main_path = "./AllLogs"
    logs = list(glob(main_path + "/*"))

    for lg in track(logs, description="Processing events..."):
        res = find_accumulators(lg)
        res = filter_params(res)
        #Console().log(sorted(res.items(), reverse=True))
        rename_logs(lg, res)
