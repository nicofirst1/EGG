#!/bin/bash
source ./args.sh


python main.py "${all_args[@]}"  --sweep_file parameters.json # # --resume_training #
