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
  "20"
  "--train_data_perc"
  "1"
  "--val_data_perc"
  "1.0"
  "--num_workers"
  "0"
  "--lr"
  "0.001"
  "--decay_rate"
  "0.8"
  "--data_root"
  "/home/dizzi/Desktop/coco/"
)

log_args=(
  "--log_dir"
  "$log_dir"
  "--tensorboard_dir"
  "$tensorboard_path"
  "--checkpoint_dir"
  "$checkpoint_path"
  "--checkpoint_freq"
  "1"
  "--train_logging_step"
  "1"
  "--val_logging_step"
  "1"

)

arch_args=(
  "--max_len"
  "6"
  "--vocab_size"
  "10"
  "--image_type"
  "seg"
  "--image_union"
  "mul"
  "--image_resize"
  "224"
  "--head_choice"
  "feature_reduction"
  "--sender_hidden"
  "128"
  "--receiver_hidden"
  "128"
  "--sender_receiver_hidden"
  "32"

)

data_args=(
  "--num_classes"
  "80"
)

loss_args=(
  "--lambda_cross"
  "1.0"
  "--lambda_kl"
  "0"
  "--lambda_f"
  "0"
  "--use_class_weights"
  "False"

)

all_args=("${train_args[@]}" "${log_args[@]}" "${arch_args[@]}" "${data_args[@]}" "${loss_args[@]}")
