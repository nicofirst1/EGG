# Intro

The goal of this project is to study how a sender and receiver behaves in a classification task with images. For more
information about the experiment results check the [Journal](markdowns/Journal.md).

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

# TODO

## Input

- Use original image with segment in sender (concat or multiply) [X]
- filter out bit/small objects [X]

## Fix bbox encodings

- convert bboxes from [x,y,h,w] (upper corner coord (x,y) and width) to center based [x_cnt, y_cnt, h,w] where now
  everything is in range 0/1

Some ideas are taken from here:

- Check [this](https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/) for possible encodings and formulas
- In end2end they [change](https://github.com/facebookresearch/detr/issues/75#issuecomment-642174524) the bbox
  representation
  with [this function](https://github.com/facebookresearch/detr/blob/be9d447ea3208e91069510643f75dadb7e9d163d/util/box_ops.py#L9)
- [Here](https://teaching.pages.centralesupelec.fr/deeplearning-lectures-build/01-pytorch-object-detection.html) they
  use the same as end2end and give
  a [custom model](https://github.com/jeremyfix/deeplearning-lectures/blob/master/LabsSolutions/01-pytorch-object-detection/models.py)
  for bboxes
- Yolo explained nicely [here](https://araintelligence.com/blogs/deep-learning/object-detection/yolo_v1/)

### Augmentation

Perform image augmentation in input. Given that we have a total of 3 different images: sender image (SI), sender
segmented (SS), receiver image (SI); we can do the following:

1. Perform same aug on all three
2. Perform same aug on sender and different receiver [X]
3. Perform all different (?)
4. Perform same SS and RI and different SI (stupid?)
5. Various combinations

For now, I would go with 1 and 2.

## Loss

- Add l1 loss to giou as in "End-to-End Object Detection with Transformers" [X]
- Why l1? Should try l2.

## Box Head

- add a trainable convolution before the flattening
- learn about the AdaptiveMaxPool2d/AdaptiveMeanPool2d [X]

#### Ref links

- Data aug for torchvision and
  coco [here](https://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detectionhttps://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detection)
- [Here](https://github.com/joheras/CLoDSA) there is a list of augs in tensorflow. Every aug that rotates/crop the image
  has to be applied to targets too.
