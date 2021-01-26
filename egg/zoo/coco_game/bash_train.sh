#!/bin/bash
source ./args.sh

python main.py "${all_args[@]}" --sweep_file parameters.json # --log_dir_uid a61c6aaa --resume_training #
