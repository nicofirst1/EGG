#!/bin/bash
source ./args.sh

all_args+=(
  "--log_dir"
  "$log_dir/Pretrain"


)

python main.py "${all_args[@]}" --sender_pretrain ./pretrain/Logs/Pretrain/b6636261 #--sweep_file parameters.json # --log_dir_uid a61c6aaa --resume_training #
