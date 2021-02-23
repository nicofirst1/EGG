# Intro

The goal of this project is to study how a sender and receiver behaves in a classification task with images. 
In the [markdown](markdowns) dir you can find more information about the code.


## Repo Structure

The structure of the repos unfolds as follows:

- [main](train.py): main python file, collects args, init classes and starts training
- [dataset](dataset.py): handle the coco dataset pipeline
- [custom logging](custom_logging.py): contains the callback necessary for the logging such as Tensorboard
- [losses](losses.py): define the custom losses used during the train
- [architectures](archs): contains the model architectures for the sender and receiver
- [utils](utils): a container of utility functions

## Setup

- Download COCO with [this script](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9) in $path2coco.

## Running

To run the training you can either use the [bash script](bash_train.sh)  or with python as:

## Bash 

The [bash script](bash_train.sh) sources the arguments from the [args script](args.sh), so be sure to launch it inside the [coco game](./) dir.
Or you can remove the source line and call it with
```shell
source args.sh
```
Once this is done you can train with 
```shell
bash bash_train.sh
```

Be sure to change the paths value inside the [args script](args.sh)

## Python 
To run with python simply use:
```shell
python train.py --data_root $path2coco
```