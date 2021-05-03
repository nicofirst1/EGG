# TODO

## Inter class uniqueness TAble

| Seed | Training Epochs | Accessories Num | Seq_Spec Seg | InterCl_uniq Seg | Seq_Spec Both | InterCl_uniq Both |
|------|-----------------|-----------------|--------------|------------------|---------------|-------------------|
| 10   | 1               | 65              | 54.8         | 1.22             | 52.2          | 2.24              |
| 11   | 1               | 75              | 57           | 3.73             | 51            | 2.56              |
| 12   | 1               | 78              | 58           | 2.25             | 52            | 1.6               |
| 22   | 1               | 74              | 55.7         | 0.771            | 52.3          | 2.156             |
| 23   | 1               | 76              | 59.1         | 2.052            | 50.9          | 2.06              |
| 19   | 2               | 150             | 49.5         | 4.9              | 43.1          | 3.47              |
| 20   | 2               | 140             | 47.4         | 3.26             | 45.2          | 3.58              |
| 21   | 2               | 158             | 49.8         | 2.67             | 42.8          | 1.99              |
| 24   | 2               | 147             | 50.6         | 4.04             | 44.2          | 3.155             |
| 25   | 2               | 145             | 47.5         | 2.033            | 45.5          | 1.94              |
| 13   | 3               | 227             | 43.3         | 3.5              | 39.5          | 5.9               |
| 14   | 3               | 212             | 43.1         | 2.88             | 43.1          | 4.55              |
| 15   | 3               | 231             | 44.8         | 4.86             | 40.1          | 5.9               |
| 26   | 3               | 222             | 43.9         | 5.95             | 41.1          | 4.07              |
| 27   | 3               | 222             | 45.6         | 4.73             | 40.7          | 3.5               |
| 16   | 5               | 384             | 38           | 8.24             | 34            | 6.19              |
| 17   | 5               | 388             | 37.6         | 6.53             | 35.9          | 8.22              |
| 18   | 5               | 365             | 37.9         | 4                | 36.6          | 6.7               |
| 28   | 5               | 416             | 37.2         | 5.9              | 38            | 4.81              |
| 30   | 5               | 389             | 38.4         | 8.9              | 35.8          | 8.06              |
| 31   | 10              | 747             | 31.9         | 7.03             | 28.9          | 9.78              |
| 32   | 10              | 740             | 29.5         | 9.94             | 28.1          | 10.19             |
|      |                 |                 |              |                  |               |                   |
|      |                 |                 |              |                  |               |                   |
|      |                 |                 |              |                  |               |                   |

| Seed | Training Epochs | Accessories Num | Seq_Spec Seg | InterCl_uniq Seg | Seq_Spec Both | InterCl_uniq Both |
|------|-----------------|-----------------|--------------|------------------|---------------|-------------------|
| Mean | 1               | 73.6            | 56.92        | 2                | 51.68         | 2.12              |
| Mean | 2               | 148             | 48.94        | 3.36             | 44.16         | 2.827             |
| Mean | 3               | 222.8           | 44.14        | 4.38             | 40.9          | 4.77              |
| Mean | 5               | 388.4           | 37.82        | 6.71             | 36            | 6.7               |
| 31   | 10              | 747             | 31.9         | 7.03             | 28.9          | 9.78              |
| 32   | 10              | 740             | 29.5         | 9.94             | 28.1          | 10.19             |
|      |                 |                 |              |                  |               |                   |

## Preprocessing

- The sender is learning to predict the classes with the receiver, it should be that the sender already know how to
  predict the classes with some degree of accuracy. So a pretraining of the sender model should be done first [X]
  
## FlatHead
- Following the papers FCOS and E2E Detection with FCN there should be at least 4 convolutional heads in the flathead
- Use sigmoid after conv [X]
- Test recevier with multiple convs



## Loss
- use focal loss [X]
- try sender class prediction 
- try sender receiver prediction 


# Old TODO
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
