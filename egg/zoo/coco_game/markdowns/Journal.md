# Take home from analysis

In the following section, we report the major discoveries coming from the analysis of the both and seg models.


## Comaprison

In this section we will analyze the main differences between the seg and both model.

### Class vs Superclass

the main differences between the models should be studied differentiating between class and superclass analysis. Indeed the per class analysis focuses more on the accuracy of each prediction, in this case all the classes are seen as diverse objects that needs to be predicted. 

On the other hand, the superclass analysis focuses more on a set of objects sharing similar characteristics. In this case it emphasize the difficulty in predicting classes with the same superclass.

TD;DR
Xclass analysis is more about accuracy, Xsuperclass is more about generalization and differentiation.

### Accuracy

Not much can be said on the accuracy level. 
Both models score pretty similarly on the class level (94.3% vs 93.4%) and on the superclass level (93.9% vs 93.3%).

### Language

Both has less symbols (854 vs 1013) than seg. Specifically the both model produced 16% of all possible combination of symbols into sequences while the seg one produced 19%.  Moreover both has a  lower average message length (5.92 vs 6), due to the occurrence of sequences with length 1 while seg sticks to the max sequence length available

Most used sequence is for the superclass accessory for both and seg.

#### Sequence specificity

Another interesting aspect of the data is seq_spec.
The sequence specificity is the proportion of sequences used mainly (more than 99% of the times) for one superclass divided by the number of sequences.
Example:
Having a sequence specificity of 10% means that 10% of all the sequences appear only with one class.

In the both model we have that 40.3% of the sequences are unique per superclass while in the seg is 44.7%. This results in more than 100 sequences which are not share in the seg model but are used in order to differentiate between input. On the other hand the both model can make this differentiation using less sequences 

#### interclass_sequences_uniqueness

The *inter class uniqueness* is calculated onto the sequence specificity. It gathers information regarding both the superclass and the class. For all the sequences in one superclass it counts the time a sequence is used with more than one class (shared).
For example a superclass with *inter class uniqueness* zero has an unique sequence for each class, while an *inter class uniqueness* of 1 means that all the sequences are shared among all the classes of a superclass.




### Corr ambiguity_rate-accuracy

The ambiguity_rate is the number of times the class happens both as a target and a distractors divided the total number
of appearances.

We see that, on the class level, this correlation is negative (-0.138) for the seg model and positive (0.017) with the
both model. This means that the the both model finds a target easier to predict the more times the class appears. Indeed
we report the same trend when it comes to the correlation between accuracy and frequency: we see that for the seg model
the correlation is (-0.005) while for both is (0.046). Altought the values are small the correlation shows that the both
model finds it easier to predict class which appear with a higher frequency.

On the other hand, when considering the superclasses, we see an opposite trend. Indeed the correlation is higher with
the seg model (-0.083) than with the both one (-0.281). Both the trends are inversly correlated, which means that the
higher the ambiguity rate the lower the cances to guess correcly. (?)

### Super class

### corr accuracy other_classes_num

On a superclass level the correlation between accuracy and the number of total classes present in the image is negative
for the seg model (-0.454) and positive for the both one (0.437). This means that the both model finds it easier to
predict a target when the image contains a lot of objects.

### corr accuracy ambiguity_rate

The ambiguity_rate is the number of times the class happens both as a target and a distractors divided the total number
of appearances.

However the correlation between ambiguity rate and accuracy is positive (0.291) for the seg model and negative (-0.259)
for the both one. This implies that the seg model finds it easier



## best take home

The seg model is better in predicting the actual target but it does so at the cost of higher sequences specialization
and number. On the other hand the both model is slightly worse at predicting the target but it share sequences across
superclasses.

# Validation problem

The problem is that the validation accuracy is much higher than the training one in most of the epochs. Following a list
of NOT possible causes:

- Resnet18: tried with Resnet50, same behavior
- Model sharing: tried initializing a resnet for both sender and receiver to no result.
- Data sharing: between train and validation there is no sharing in terms of images or annotations
- Number of distractors: same behavior can be seen with distractors = 1, 2, 3
- Target predictability: there is no predictable pattern for the location of the target between train and val or between
  the same dataset and multiple epochs.
- Removing MeanBaseline: using the NoBaseline instead does not alleviate the pattern, but it does slow down the
  learning.
- Reduce train samples to match val samples: the pattern still persists when len(val)/len(train)= 87% and 210%
- Use val to train and train to val: the pattern persist, showing that is not a data problem.
- RL Optimizer: not it
- Train eval mode: correct for the phase.
- Loss.backward/optimizer.step: maybe the loss.backward is called between the end of the training and the start of the
  validation -> Nope
- Removed filtering: class filtering based on number of distractors is not the culprit
- Random item deletion: nope
- Change sender image process: same pattern appears for the first 3 epochs, then train>val.
- Do only val: obv not train, stuck at 50%
- First val then train: Nope
- Drop last batch = True: Nope
- Change data seed: Nope
- Debug with batchsize = 16: not learnig, the first three epochs the val seems to learn then goes back to 50%
- add CenterCrop to transformation: maybe this is it? Nope it got thing worse
- Could be feature extraction Sender/Receiver: check with same image in train eval if features are the same for both:
  they are the same...

Todo:

- Change receiver head: signal_expansion same thing; only_image not learning
- ?

## Observations

Some observations on the pattern:

- The same pattern occurs when the validation data is set to the train data. That is the two datasets are identical.
- The same pattern occurs when the validation data is taken as a split from the train data (20%). In this instance the
  two dataset are not identical.
- The pattern does NOT occur when the validation data is replaced with a dummy dataset. The dummy dataset is created at
  random and has no (visual) meaning what so ever.
- The gap between validation and train accuracy decreases the more epochs there are.
- The system does not learn with 20 samples... i was expecting overfitting but nope
- Datasamples: with 20 samples the first two trains follows the same patter as always

Observation on the other metrics:

- Loss starts low (0.5), then increases during ~4 epochs (1.4), during this time train loss < val loss. then decreases
  again with train loss > val loss.

- X loss: decreases steadily during the train

# Experiments Discrimination

These esperiments were ran with 80 classes from the coco dataset on the discrimination task. The sender is shown the
complete image and a segment of an object inside the image. The receiver is shown the segment plus a number of
distractors which are segments of objects coming from the same image.

## Architectures

### Sender

The sender stays similar to the previous version (please check the *Image manipulation*  chapter in the **Experiments
Classification** section ). A new introduction is a linear layer which takes as input the sender hidden state (the one
which will be later passed to the Reinforce wrapper) and performs a prediction in the output size. This was created for
the classification task.

### Receiver

The receiver was heavily modified. It now takes as input the sender signal as previously and N (= distractors +1)
images. The output must be of size N indicating the segment shown to the sender.

The receiver architectures are currently 3 + 4 control ones:

- FeatureReduction: similar to the default discrimination architecture, the feature reduction embeds the vision feature
  to be of equal size to the signal length. It then performs a multiplication between the two matrices of
  size   `[batch size, N, signal length] x  [batch size, signal length, 1]`, which then yields a class logit of size
  `[batch size, N, 1]`

- SignalExpansion: contrary to the previous, the signal expansion embeds the signal dimension to match the vision one
  and repeats the matrix multiplication above.

- Conv: Similarly to FeatureReduction, Conv reduces the dimension of the vision to the signal one.
  `[batch size, N, vision features] ->  [batch size, N, signal length]`. It then repeats the signal on the second
  dimension to match the vision one `[batch size, signal length] ->  [batch size, N, signal length]`. This allow for the
  matrix multiplication between the signal and the vision vector. The output's dimension is increased to reach 4 (min
  output for conv2d)
  `[batch size, N, signal length, 1]`. Finally, the convolution can be applied in order reduce the second dimension
  from `N` features to `1`  feature and a non-squared kernel of size` (signal length - N,1)`
  reduces the last dimension to the output. `[batch size, N, signal length, 1] ->[batch size, 1, N, 1]`
  The result is squeezed to get a class logits of size
  [batch size, N]

The control ones are:

- Only signal: the receiver uses only the signal to perform the prediction. It is now more difficult to guess using only
  the signal since the sender has no information on the amount or position of the distractors.

- Random signal: same as before, but the signal is randomized.

- Only image: similar to only signal, but the receiver uses the images only to perform the prediction.
- Random signal image: the receiver uses both the signal and the image as in FeatureReduction, but the signal is
  randomized.

## Results

coming soon

# Experiments Clasification (old)

These experiments were ran with 15 classes from the coco dataset (skipped first 5) on the classification task. The
performance was measured by the test class accuracy. Whenever the `***` is present then more experiments should be done.

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

## Pretrained

In this setting the pretrained module is taken into consideration. Such modules trains the sender on the classification
task using an additional linear layer of size [hidden dim, num classes] which is basically a map from the internal
hidden size (which will later form the message) and the class.

As for the first result: training the sender for 40 epochs achieves an accuracy of 75% with conventional architectures.
Although a slight increase can be observed when using `image_type=both & image_union= cat` the `image_union = mul` has
been chosen since it performs better in cooperation with the receiver.

Using this pretrained sender vs a new one achieves better scores on the sender-receiver game (+4%) bringing the receiver
accuracy to a stable 72% (the highest value so far). The information coming from the sender class accuracy further
highlights this difference. Indeed when the sender has not been pretrained on the task its accuracy does not surpass the
pure random change (7% with 15 classes). On the other hand it must be said that no training is performed in order to
increase such accuracy ***.

## Flat Head Choice

The flat head choice refers to the class in [flats](../archs/flats.py). The most basic version is a simple average
pooling which takes as input the 3D data coming from the pretrained vision model and performs an average poo (as if it
was the last original layer of resnet).

Most of the experiments were done in the pretrain framework, where the sender is pretrained on the classification task
without the receiver, but some were performed on the communication task too.

### Pretrain Framework

Various architectures were tested in this framework, for everyone the feature size was 256 (more experiments with
different sizes are necessary ***), for those with more than one conv the stride increased in the following way 1,2,3,4:

- Simple convolutions of various depth: in this category the networks are simple convolution one after the other such as
  Conv1/Conv2/Conv3/Conv4
- Convolution with batch norm: adds a batch normalization after each convolution Conv1BatchNorm
- Convolution with sigmoid: as reported in the paper FCOS, a sigmoid operation after the convolution helps with the
  classification task.
- Other types of arch aimed at understanding the relevance of some parameters

From the experiments it is clear that adding a sigmoid helps with the overall performance (+3%). No significant
improvement is seen with more convolutions, so keeping the number to one both reduces the total parameters and the
training time. Batch normalization does not help even when multiple convolutions are present, moreover the combination
of batch norm, and a sigmoid layer yields worse result than the two separate (it is important to notice how the agent
performs better when the sigmoid is place after the batch norm). Finally, keeping the stride constant yields the worst
results so far.

Optimal Value: Single convolution with sigmoid

### Communication Framework

Experimenting with a reduced set of archs on the communication framework resulted in:

- Random classification when using the conv1Sigmoid on the sender without pretraining
- +6% when the sender uses average pool and the receiver Conv1Sigmoid
- Best results so far (+2% ) with a pretrained sender , independently of the receiver arch.

On the other hand it is interesting to notice that the sender accuracy does not reach the performance yielded in the
pretraining phase (-30%). Indeed, the sender performance does not even start from the previous one. It must be specified
that there is no optimization in respect to the sender performance in this framework.

