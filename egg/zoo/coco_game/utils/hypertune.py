import argparse
import itertools
import json
import sys

from egg.zoo.coco_game.utils.utils import console


def product_dict(**kwargs):
    """
    Get combinations of parameters
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def set_sys_args(params):
    """
    Set sys.argv with given parameters
    """
    # save functiun call
    first = sys.argv[0]

    # transform from dict to list of --key value
    params = [[f"--{k}", v] for k, v in params.items()]
    params = [x for sub in params for x in sub]

    # check if there are any bool
    assert not any(isinstance(x, bool) for x in params), "Cannot pass Bool values when action store is true"

    # cast to string
    params = [str(x) for x in params]
    # check for args not present in params and save them
    not_set_args = [x for x in (set(sys.argv) - set(params)) if "--" in x]
    values = [sys.argv[sys.argv.index(x) + 1] for x in not_set_args]

    # remove values captured for args with no value
    no_value=[values.index(x) for x in values if "--" in x]
    for idx in no_value:
        not_set_args.pop(idx)
        values.pop(idx)


    to_add = list(sum(zip(not_set_args, values), ()))
    to_add.insert(0, first)
    # define new sys.argv
    sys.argv = to_add + params


def hypertune(main_function):
    """
    Decorator performing hypertune on parameters
    """

    sweep_file_name = "sweep_file"
    sweep_file_arg = f"--{sweep_file_name}"

    # parse params
    parser = argparse.ArgumentParser()
    parser.add_argument(
        sweep_file_arg,
        help="Path to json parameters file",
    )
    args, _ = parser.parse_known_args()

    if args.__getattribute__(sweep_file_name) is not None:
        # extract json
        with open(args.__getattribute__(sweep_file_name)) as json_file:
            parmas = json.load(json_file)

        # get combination generator
        combinations = list(product_dict(**parmas))

        # remove sweep_file_arg from sys arg
        index = sys.argv.index(sweep_file_arg)
        sys.argv.pop(index)
        sys.argv.pop(index)

        console.log(f"There are {len(combinations)} combinations")

        # iterate over possible combinations
        for p in combinations:
            set_sys_args(p)
            main_function()

        console.log("HyperParameter search completed")
    else:
        main_function()

    sys.exit(0)
