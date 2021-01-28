#!/bin/bash
source ./args.sh

all_args+=(
  "--log_dir"
  "$log_dir/Other"


)

python main.py "${all_args[@]}"  --sweep_file parameters.json #--sender_pretrain ./pretrain/Logs/Pretrain/b6636261 # --resume_training #
