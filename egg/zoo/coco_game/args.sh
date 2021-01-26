#!/bin/bash
checkpoint_path="checkpoints"
interactions_path="interactions"
tensorboard_path="tensorboard"
log_dir="Logs"
home="/home/dizzi/Desktop/EGG/"

export PYTHONPATH="$home/egg/zoo/coco_game/:$home"

train_args=(
  "--batch_size"
  "64"
  "--n_epochs"
  "4"
  "--train_data_perc"
  "1"
  "--test_data_perc"
  "1.0"
  "--num_workers"
  "4"
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
  "--test_log_prob"
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
  "50"
  "--test_logging_step"
  "5"
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
  "64"
  "--receiver_hidden"
  "64"
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
  "15"
  "--skip_first"
  "5"
)

loss_args=(
  "--cross_lambda"
  "1.0"
  "--kl_lambda"
  "1.0"
  "--use_class_weights"
  "True"

)

all_args=("${train_args[@]}" "${log_args[@]}" "${arch_args[@]}" "${data_args[@]}" "${loss_args[@]}")
