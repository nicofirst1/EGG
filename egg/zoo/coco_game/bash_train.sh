#!/bin/bash
source ./args.sh

python train.py "${all_args[@]}" --sweep_file parameters.json  # --resume_training #
