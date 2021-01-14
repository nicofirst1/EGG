import json
from glob import glob

from rich.progress import track
from rich.console import Console
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
        ['num_workers', "tensorboard_dir", "random_seed", "resume_training", "checkpoint_dir"])
    diffs -= to_remove

    for k, v in res_dict.items():
        res_dict[k] = {k2: v2 for k2, v2 in v.items() if k2 in diffs}

    return res_dict


def find_accumulators(path2logs, tag="test/class_accuracy"):
    logs = list(glob(path2logs + "/*"))

    res = {}
    for idx in track(range(len(logs)), description="Processing events..."):
        log = logs[idx]
        event_path = log + "/tensorboard/events*"
        try:
            events = sorted(glob(event_path))[-1]
        except IndexError:
            continue
        ac = get_accumulator(events, tag)

        if ac is None:
            continue
        else:
            ac *= 100
        params = get_params(log)
        params[tag] = ac
        params['log_dir_uid'] = params['log_dir_uid'].split("/")[-1]
        res[ac] = params

    return res


if __name__ == '__main__':
    res = find_accumulators("Logs")
    res = filter_params(res)
    Console().log(sorted(res.items(),reverse=True))
