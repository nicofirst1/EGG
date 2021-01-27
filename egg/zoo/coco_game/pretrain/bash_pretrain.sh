#!/bin/bash
source ../args.sh

# modify some args ( be sure to not touch anything related to the sender architecture!)
all_args+=(
  "--log_dir"
  "Logs/Pretrain"
  "--n_epochs"
  "10"


  )

python pretrain.py "${all_args[@]}" #--sweep_file ../parameters.json # --log_dir_uid a61c6aaa --resume_training #
