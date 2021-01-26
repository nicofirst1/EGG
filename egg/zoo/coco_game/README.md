# Intro

The goal of this project is to study how a sender and receiver behaves in a classification task with images. 
In the [markdown](markdowns) dir you can find more information about the code.


## Repo Structure

The structure of the repos unfolds as follows:

- [main](main.py): main python file, collects args, init classes and starts training
- [dataset](dataset.py): handle the coco dataset pipeline
- [custom logging](custom_logging.py): contains the callback necessary for the logging such as Tensorboard
- [losses](losses.py): define the custom losses used during the train
- [architectures](archs): contains the model architectures for the sender and receiver
- [utils](utils): a container of utility functions

## Setup

- Download COCO with [this script](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9) in $path2coco.

## Running

To run the training you can either use the [bash script](train.sh) by changing the parameters inside the script or with
python as:

```
python main.py --data_root $path2coco
```
