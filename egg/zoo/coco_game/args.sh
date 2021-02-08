#!/bin/bash
checkpoint_path="checkpoints"
tensorboard_path="tensorboard"
log_dir="Logs"
home="/home/dizzi/Desktop/EGG/"

export PYTHONPATH="$home/egg/zoo/coco_game/:$home"

train_args=(
  "--batch_size"
  "128"
  "--n_epochs"
  "4"
  "--train_data_perc"
  "1"
  "--val_data_perc"
  "1.0"
  "--num_workers"
  "8"
  "--lr"
  "0.001"
  "--decay_rate"
  "0.8"
  "--data_root"
  "/home/dizzi/Desktop/coco/"
)

log_args=(
  "--train_log_prob"
  "0.0"
  "--val_log_prob"
  "0.0"
  "--log_dir"
  "$log_dir"
  "--tensorboard_dir"
  "$tensorboard_path"
  "--checkpoint_dir"
  "$checkpoint_path"
  "--checkpoint_freq"
  "1"
  "--train_logging_step"
  "300"
  "--val_logging_step"
  "30"
  "--use_rich_traceback"

)

arch_args=(
  "--max_len"
  "3"
  "--vocab_size"
  "20"
  "--image_type"
  "both"
  "--image_union"
  "mul"
  "--image_resize"
  "224"
  "--head_choice"
  "simple"
  "--sender_hidden"
  "128"
  "--receiver_hidden"
  "128"
  "--receiver_num_layers"
  "1"
  "--sender_num_layers"
  "1"
  "--box_head_hidden"
  "32"
  "--sender_embedding"
  "16"
  "--receiver_embedding"
  "16"
  "--sender_cell_type"
  "gru"
  "--receiver_cell_type"
  "gru"
)

data_args=(
  "--min_area"
  "0"
  "--num_classes"
  "80"
  "--skip_first"
  "0"
)

loss_args=(
  "--lambda_cross"
  "1.0"
  "--lambda_kl"
  "0"
  "--lambda_f"
  "0"
  "--use_class_weights"
  "True"

)

all_args=("${train_args[@]}" "${log_args[@]}" "${arch_args[@]}" "${data_args[@]}" "${loss_args[@]}")
