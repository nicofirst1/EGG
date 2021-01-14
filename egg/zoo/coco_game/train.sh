#!/bin/bash
checkpoint_path="checkpoints"
interactions_path="interactions"
tensorboard_path="tensorboard"
log_dir="HyperLogs"
home="/home/dizzi/Desktop/EGG/"

export PYTHONPATH="$home/egg/zoo/coco_game/:$home"

train_args=(
  "--batch_size"
  "64"
  "--n_epochs"
  "18"
  "--train_data_perc"
  "1"
  "--test_data_perc"
  "1.0"
  "--num_workers"
  "8"
  "--lr"
  "0.0001"
)

log_args=(
  "--train_log_prob"
  "0.0001 "
  "--test_log_prob"
  "0.005"
  "--log_dir"
  "$log_dir"
  "--tensorboard_dir"
  "$tensorboard_path"
  "--checkpoint_dir"
  "$checkpoint_path"
  "--checkpoint_freq"
  "2"
  "--train_logging_step"
  "50"
  "--test_logging_step"
  "5"
  "--use_rich_traceback"

)

arch_args=(
  "--max_len"
  "2"
  "--vocab_size"
  "5"
  "--image_type"
  "img"
  "--image_union"
  "mul"
  "--image_resize"
  "224"
  "--head_choice"
  "single"
  "--sender_hidden"
  "16"
  "--receiver_hidden"
  "16"
  "--receiver_num_layers"
  "1"
  "--sender_num_layers"
  "1"
  "--box_head_hidden"
  "16"
  "--sender_embedding"
  "16"
  "--receiver_embedding"
  "16"
  "--sender_cell_type"
  "lstm"
  "--receiver_cell_type"
  "lstm"
)

data_args=(
  "--min_area"
  "0"
  "--num_classes"
  "15"
)

loss_args=(
  "--cross_lambda"
  "1.0"

)

all_args=("${train_args[@]}" "${log_args[@]}" "${arch_args[@]}" "${data_args[@]}" "${loss_args[@]}")

python main.py "${all_args[@]}" #--log_dir_uid 597c9406 --resume_training
