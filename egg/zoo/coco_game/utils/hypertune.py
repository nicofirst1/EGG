import argparse
import itertools
import json
import sys

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def hypertune(main_function):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param_json",
        help="Path to json parameters",
    )
    args, _ = parser.parse_known_args()
    with open(args.param_json) as json_file:
        parmas=json.load(json_file)

    combinations=product_dict(**parmas)

    # remove param json from sys arg
    index=sys.argv.index("--param_json")
    sys.argv.pop(index)
    sys.argv.pop(index)
    for p in combinations:
        a=1
        main_function()



