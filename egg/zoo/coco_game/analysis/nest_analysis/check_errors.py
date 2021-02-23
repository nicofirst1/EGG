from collections import Counter

from tensorboard.backend.event_processing import event_accumulator

from egg.zoo.coco_game.analysis.nest_analysis import path_parser


def parse_error_args(nest_path, idxs):
    confs = []
    for idx in idxs:
        out_file = list(nest_path.rglob(f"{idx}*.out"))[0]
        with open(out_file, "r") as f:
            lines = f.readlines()

        lines = "\n".join(lines)
        confs.append(lines)

    confs = [x.split("[", 1)[1].split("]", 1)[0] for x in confs]
    confs = [x.split(",") for x in confs]

    res = {}

    for c in confs:

        for p in c:

            if "=" not in p: continue

            key = p.split("=")[0].replace("--", "")
            val = p.split("=")[1]

            if key not in res.keys():
                res[key] = []

            res[key].append(val)

    for k, v in res.items():
        res[k] = Counter(v)

    return res


def parse_error_ids(nest_path, err_string="RuntimeError: DataLoader timed out"):
    errors_files = nest_path.rglob("*.err")

    ids = []
    for e in errors_files:
        with open(e, "r") as f:
            lines = f.readlines()

        lines = "\n".join(lines)

        if err_string in lines:
            ids.append(e.name)

    ids = [x.split("_")[0] for x in ids]
    return ids


def parse_uuids(nest_path, ids):
    uuids = []
    for idx in ids:
        out_file = list(nest_path.rglob(f"{idx}*.out"))[0]
        with open(out_file, "r") as f:
            lines = f.readlines()

        lines = [x for x in lines if "log_dir_uid" in x][0]
        # clean line
        lines = lines.strip().split(",")[1].replace("'", "").replace(")", "")
        uuids.append(lines)
    uuids = [x.strip() for x in uuids]
    return uuids


def get_accumulator(path2accumulator, tag):
    size_guidance = {  # see below regarding this argument
        event_accumulator.COMPRESSED_HISTOGRAMS: 0,
        event_accumulator.IMAGES: 0,
        event_accumulator.AUDIO: 0,
        event_accumulator.SCALARS: 10,
        event_accumulator.HISTOGRAMS: 0,
    }

    ea = event_accumulator.EventAccumulator(str(path2accumulator), size_guidance)
    ea = ea.Reload()
    try:
        ep = ea.Scalars(tag)
    except KeyError:
        ep = []
    return ep


def parse_epochs(nest_path, uuids):
    log_paths = list(nest_path.rglob(f"Logs/Both/*"))
    error_paths = [x for x in log_paths for y in uuids if y in x.name]

    log_paths = set(log_paths) - set(error_paths)
    log_paths = list(log_paths)

    error_epochs = []
    for lp in error_paths:
        lp = list(lp.rglob(f"runs/events*"))[0]
        error_epochs.append(get_accumulator(lp, "epoch"))

    standard_epochs = []
    for lp in log_paths[:10]:
        lp = list(lp.rglob(f"runs/events*"))[0]
        standard_epochs.append(get_accumulator(lp, "epoch"))

    return error_epochs, standard_epochs


def parse_mean_time(error_epochs, standard_epochs):
    error_times = []

    for epochs in error_epochs:
        mean_time = 0
        for idx in range(len(epochs) - 1):
            ep1 = epochs[idx]
            ep2 = epochs[idx + 1]
            time = ep2.wall_time - ep1.wall_time
            total_ep = ep2.value - ep1.value
            time /= total_ep
            mean_time += time
        mean_time /= len(epochs) - 1
        error_times.append(mean_time)

    standard_times = []

    for epochs in standard_epochs:
        mean_time = 0
        for idx in range(len(epochs) - 1):
            ep1 = epochs[idx]
            ep2 = epochs[idx + 1]
            time = ep2.wall_time - ep1.wall_time
            total_ep = ep2.value - ep1.value
            time /= total_ep
            mean_time += time
        mean_time /= len(epochs) - 1
        standard_times.append(mean_time)

    error_times = sum(error_times) / len(error_times)
    standard_times = sum(standard_times) / len(standard_times)

    return error_times, standard_times


if __name__ == '__main__':
    nest_path = path_parser()

    # parse error configurations
    # ids = parse_error_ids(pt)
    # res = parse_error_args(pt, ids)

    ids = parse_error_ids(nest_path, err_string="oom-kill event")
    uuids = parse_uuids(nest_path, ids)
    error_epochs, standard_epochs = parse_epochs(nest_path, uuids)
    error_times, standard_times = parse_mean_time(error_epochs, standard_epochs)

    print(f"Mean time for one epoch on oom error : {error_times}, while standard is : {standard_times}")
