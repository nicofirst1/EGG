import argparse
import itertools
import json
import sys

from egg.zoo.coco_game.utils.utils import console


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def set_sys_args(params):
    first = sys.argv[0]
    params = [[f"--{k}", v] for k, v in params.items()]

    params = [x for sub in params for x in sub]

    assert not any(isinstance(x, bool) for x in params), "Cannot pass Bool values when action store is true"
    params = [str(x) for x in params]

    not_set_args = [x for x in (set(sys.argv) - set(params)) if "--" in x]
    values = [sys.argv[sys.argv.index(x) + 1] for x in not_set_args]
    to_add = list(sum(zip(not_set_args, values), ()))
    to_add.insert(0, first)
    sys.argv = to_add + params


def hypertune(main_function):

    sweep_file_arg="--sweep_file"

    # parse params
    parser = argparse.ArgumentParser()
    parser.add_argument(
        sweep_file_arg,
        help="Path to json parameters file",
    )
    args, _ = parser.parse_known_args()

    # extract json
    with open(args.param_json) as json_file:
        parmas = json.load(json_file)

    # get combination generator
    combinations = product_dict(**parmas)

    # remove sweep_file_arg from sys arg
    index = sys.argv.index(sweep_file_arg)
    sys.argv.pop(index)
    sys.argv.pop(index)

    console.log(f"There are {len(combinations)} combinations")
    #iterate over possible combinations
    for p in combinations:
        print(sys.argv)
        set_sys_args(p)
        print(sys.argv)

        main_function()
    console.log("HyperParameter search completed")
