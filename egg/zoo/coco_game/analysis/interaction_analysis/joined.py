import csv
import json

from egg.zoo.coco_game.analysis.interaction_analysis.utils import (
    console,
    path_parser, split_line, define_out_dir,
)


def get_hierarchy_classes(lines):
    class_hierarchy = {}
    for l in lines:
        data = split_line(l)

        tc = data['target_class']
        tsc = data['target_superclass']

        if tsc not in class_hierarchy.keys():
            class_hierarchy[tsc] = {}

        if tc not in class_hierarchy[tsc].keys():
            class_hierarchy[tsc][tc] = 0



        class_hierarchy[tsc][tc] += 1

    for k, v in class_hierarchy.items():
        total = sum(v.values())

        for k2, v2 in v.items():
            class_hierarchy[k][k2] = v2 / total

        class_hierarchy[k]['total'] = total

    total = sum([class_hierarchy[x]['total'] for x in class_hierarchy.keys()])

    for k in class_hierarchy.keys():
        class_hierarchy[k]['total_perc'] = class_hierarchy[k]['total'] / total

    return class_hierarchy


def joined_analysis(interaction_path, out_dir):
    console.log("Computing joined analysis...")

    with open(interaction_path, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)

    header = lines.pop(0)
    assert len(lines) > 0, "Empty Csv File!"

    ch = get_hierarchy_classes(lines)
    ch_path = out_dir.joinpath("joined_class_hierarchy.json")

    with open(ch_path, "w") as f:
        json.dump(ch, f)

    return dict(
        joined_class_hierarchy=ch,
    )


if __name__ == "__main__":
    interaction_path = path_parser()
    out_dir = define_out_dir(interaction_path)

    joined_analysis(interaction_path, out_dir)
