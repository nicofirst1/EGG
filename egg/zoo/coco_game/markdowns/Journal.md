# Experiments

These experiments where run with 15 classes from the coco dataset (skipped first 5)
The performance was measured by the test class accuracy. Whenever the `***` is present then more experiments should be
done.

## Hidden size

In this part the variation of the parameters `sender/receiver_hidden_size` will be studied. This parameter controls the
two different things:

    1. In the "RnnReceiverDeterministic" is the hidden dimension of the `RnnEncoder` which translates to the dimensions of the chosen rnn cells
    2. In the "RnnSenderReinforce" the dimension controls the hidden to output linear module

Both are directly tied to the deepness of the language interpretation part.

The parameters were changed together from a min of 16 to a max of 256. Moreover, two runs were executed per each
configuration in order to get a better generalization, the variance observed between runs with the same configuration is
in range [2,12]%

Overall the best results where given by a value of 128. The trends seems to follow a linear fashion where the increase
in the hidden number maps to an increased performance. This is not true for the 256 value which yielded worst results
than 64.

Optimal values: `sender_hidden_size:128` `receiver_hidden_size:128`

## Vocab

The vocabulary size and length (`vocab_size, max_len`) control the message structure and are the most influential
parameters so far. Again there seems to be a linear relationship between the `vocab_size` and the accuracy which
increases steadily until a value of 30 (15 classes). For greater values (>50) the performances see a drastic drop (30%).
On the other hand the `max_len` seems to be less influential as long as the product of its value and the `vocab_size` is
kept under 40 with `vocab_size>10`. Practically speaking the language variance defined by the `vocab_size` parameter
seems to be the main factor contributing to a good performance which can deal with short message sequences. On the other
hand the agents appear to be confused when the `max_len` increases too much.

Optimal values: `vocab_size:20` `max_len:3`

## Optimizer

As for now there are 3 available optimizers:

1. Adam 2. SGD 3. AdaGrad

The results show that adam is in general better since it starts from a high value (+40% than others) but the learning
stop shortly after gaining +10 % On the other side, a much longer training was performed with the SGD which yielded
similar results to adam in x25 epochs.

Optimal value : adam

## Cell Types

In this part the parameters `sender/receiver_cell_type` were varied independently. From the results it emerges how the
value `sender_cell_type=gru` seems to influence the outcome the most (+4%). On the other hand the `receiver_cell_type`
does not seem to be as influential but performs best when not `=rnn`.

Optimal values: `sender/receiver_cell_type=gru`

## Loss

In here the parameters `cross/kl_lambda` and `use_class_weights` where modified. It is clear that kl loss cannot
compensate for the cross entropy, indeed when `cross_lambda=0` no values for the other two parameters can compensate a
loss in 40% performance. On the other hand `kl_lambda` seems to help slightly (3%) when present with a linear
contribution.

Moreover, when setting `use_class_weights=True` a slight increase in accuracy (+2%) is achieved.

Optimal value : `cross_lambda=1` & `kl_lambda=1.0` & `use_class_weights=True`

## Number of Layers

In this part the parameters `sender/receiver_num_layers` were tested. These parameters represent the number of
recursions a rnn cell (independently of the type) has.

From the experiments it is clear that any value >1 comes with a drop in performance. More tests should be performed with
more epochs to see if the performance can increase ***,

## Image size

The pretrained vision model accepts images of size > 224. The size of the image is directly correlated to the speed of
the training phase (+10 px = + 70 s). Overall there doesn't seem to be a linear relationship between the performance,
and the image size, so the fastest (224) was chosen.

Optimal value: `image_resize =224`

## Image manipulation

The sender gets as input the entire image and a segment in which the object is presented, how it will use this
information is dictated by the parameters `image_type` and `image_union` as follows:

- `image_type= img`: the sender considers only the image
- `image_type= seg`: the sender considers only the segment
- `image_type= both` & `image_union=mul` : the sender considers both the images and get the product between the features
  extracted by both
- `image_type= both` & `image_union=cat` : the sender considers both the images and get the concatenation between the
  features extracted by both

As can be seen when `image_type!= both` the `image_union` parameter is discarded.

From the experiments it is clear what follows:

- Using the product of the two features embeddings yields the best results so far
- Using only the segment or the image has worse performances (-10%), no significant change can is reported between 'img'
  and 'seg'.
- Using the concatenation brings the accuracy near random level (-40%)

Optimal value: `image_type= both` & `image_union=mul`

## Head Choice

The receiver gets as input both the features-image and the (encoded)-message from the sender, these to inputs are passed
to the receiver head which can do the following depending on the value of `head_choice`.

### Available heads

First lets consider the cases in which only one of the two inputs is regarded:

- `head_choice= only_signal` : in this case only the signal is passed to a linear layer of size (signal dim, num
  classes)
- `head_choice= only_image` : in this case only the signal is passed to a linear layer of size (features dim, num
  classes)

When both inputs are considered, multiple ways to merge them are available:

- `head_choice= signal_expansion_*` : in this case the signal is passed to a linear layer (signal dim, features dim)
  which expands its dimensions to match the image-features one
- `head_choice= feature_reduction_*` : in this case the signal is passed to a linear layer (features dim, signal dim)
  which reduces the image-feature dimensions to match the signal ones

These two methods then are split by the type of merging done between the single and feature

- `head_choice= *_mul` : the two vectors are multiplied together
- `head_choice= *_cat` : the two vectors are concatenated together

Moreover, other implementations are available which consider both the inputs:

- `head_choice= simple` : cat together signal and image-features and pass them to a linear (signal_dim + feature_dim,
  num classes)
- `head_choice= sequential` : cat together signal and image-features and pass them to a sequential model
  with [nn.Linear(
  signal_dim + feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(), nn.Linear(hidden_dim, num_classes)]

Finally, in order to test random configurations, two additional heads are provided:

- `RandomSignal`: keeps the same architecture as `only_signal` but generates a random signal instead
- `RandomSignalImg`: uses the same arch as `simple` but with a randomly generated signal.

### Results

First we must say that the results didn't show a significant variance between the best and the worst (+10%). This being
said the best results were given by the product `*_mul` of the image-features times the signal in both the signal
expansion and feature reduction cases. On the other hand the `*_cat` models performed worse

Moreover, while the `only_singal` and `simple` yielded similar results, it is important to higlight that, during
training, the model with `only_signal` shows a sharp increase in the first epochs (+40%) starting from a random
baseline, while the `simple` model takes far less time to converge to the same value. This shows how the model with no
image information has to rely solely on the signal, on the other hand the model using both has no need for redundant
information about the image in the sender message.

The `*_cat` models performed worse than the `*_mul` ones achieving a similar results to the `sequential` model.

On the other hand, `RandomSignal` achieves the same results as blindly guessing the (biased) class (12% with 15 classes)
, while `RandomSignalImg` yields similar results to `only_image` defined below.

Finally, the `only_image` model which discards completely the message sent from the receiver achieves +50% than the
random signal. This result is fundamental since it shows that without the sender message the receiver is still able to
correctly classify the image with a 60% accuracy

Optimal value: `head_choice= signal_expansion_mul`

## Embeddings ***

todo
