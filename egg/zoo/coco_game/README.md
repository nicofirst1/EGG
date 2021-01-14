## Setup

- Download COCO with [this script](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9) in $path2coco.

## Running

To run the trainig use:

```
python main.py --root $path2coco
```

My training paramteres:

```
python main.py --batch_size 64 --max_len 10\
--train_data_perc 0.25 --image_type both\
--image_union cat --simple_receiver\
--train_log_prob 0.001 --test_log_prob 0.05\
--checkpoint_dir ./checkpoints --checkpoint_freq 1\
--tensorboard --resume_training\
--giou_lambda 0.7 --l1_lambda 1.0
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

## Ideas

### Game mechanics

- Multi step game (task & talk), allow receiver to ask questions and sender to answer
- Action aid: gridworld where floor is image, receiver needs to move to a location. The location is described by the
  sender. In this you can add competitiviness ( first receiver to get to goal wins).
- Receiver outputs the an image filter which applied to the original image would have yielded the best results for him.
  This filter is then passed to the sender which :
    - Uses it on the original image and outputs another message. This chain can repeat until a max is reached
    - Uses it as supervised learning and tries to replicate it. During test it could try to replicate it, apply it on
      the image and generate a message based on it
    - For multi-step game use transformers where sequence is history of messages (both sender receiver of just self?)

### Input

- Use video
- Use biased test/train
    - Bias on objects dimensions
    - Bias on object location
    - Bias on object colors (?)
    - Bias on object category

- Reduce number of classes

### Modeling others

- model receiver in sender, use input + message and receiver out
- Switch roles using model of other agent (selfplay inter-agent?)
- Model in sender and influence
- Adversarial sender: models receiver and outputs the worst message.

### Population/ Iterating learning

- Randomly change receiver/sender
- Use generation learning
- Chain of receivers (gioco del telefono): a first sender describes the object, the embedding is passed to a chain of N
  receivers. Each one tries to guess (sender gets total (inversed?) discounted reward ) and pass guess to next one. You
  can also try with receiver speaking different languages and use machine translation as loss.
- Train on two on input perturbation two noraml and confront

### Other

- Try Gumbel
- Adversarial somewhere?
- Inverse vision system from message (and image?), could be used as aid for receiver or as adversarial.
- Use spoken language?
- homonomy filter