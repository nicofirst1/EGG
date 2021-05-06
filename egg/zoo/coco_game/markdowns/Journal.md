# Observations

## Class
### Seg



- Accuracy is negatively correlated with the ambiguity rate (-0.314): the more a class appears both as target and distractor the less accuracy.



- Ambiguity rate is high pos (0.96) with amb richness: classes with more target==distr have more sequences.

-  Class richness is high pos (0.88) to the class frequency: more frequent classes have more symbols/sequences
-  Target freq is high neg (-0.53) to frequency: classes which are more frequent appear less as a target and more as other classes 

### Differences with both 

First of all seg has more sequences than both (1360 vs 1135 ) normalized by the max combinations of sequences 531630 (0.21% vs 0.26%).

Seg has higher values for all types of specificity, which means that it has more sequences which are used once for class/superclass. Instead both has a lower number of sequences overall which are shared more across objects.

### Difference Correlation 


- Correlation between *X_trg==dstr* and *othr_clss_num* is 0.043 for seg and 0.266 for both (0.043 p).
The correlation measure how difficult is the prediction in relation with the number of classes present in the overall image. Since the Seg model does not have access to the image the correlation is practically zero, while for the both model is positive. This means that both finds it more difficult to distinguish target from distractor (when they are the same class) when there is a high number of objects in the image.


- Correlation between *othr_clss_num* and *prec_sc* is  0.077 on seg and  -0.303 for both (0.0176 p).
The correlation measures the precision in correctly predicting a class when target==distractor in relationship with the number of objects in the image. Here too we see how the both model is confused by number of object in the image while the seg one nope.


- On the other hand, correlation between *othr_clss_num* and *prec_oc* is   -0.242 on seg and  0.024  for both ( p>0.05) .
The correlation measures the precision in correctly predicting a class when target!=distractor in relationship with the number of objects in the image. 
In a completely different results as before, here we see how the number of objects in the image influences negatively Seg when it needs to differentiate between to objects coming from different classes.


- Correlation between *freq* and the two precisions *prec_oc*  ( p>0.05) and *prec_sc* (0.0194 p) confirms what previsouly said. Indeed for the Seg model this correlation is negative with *prec_oc* (-0.251) and zero with *prec_sc* (0.049), on the other hand, Both has a negative correlation with *prec_sc* (-0.299) and zer with *prec_oc* (0.041).
This means that Seg is fooled in predicting targets which appears frequently when the distractor is not the same class, on the other hand both is fooled when the opposite is true.

TL;DR
Seg is better at differentiating between objects with the same class while Both is better when target!=distractor.


- Correlation between *trg_freq* and *intracls_seq_spcf* is  -0.356 on seg and   0.015  for both  (0.0061 pval).
The correlation measures how many unique senquencies a classes has in relationship with its target frequency. Since the *intracls_seq_spcf* is multiplied by the *trg_freq* in its computation we actually have a relationship between the number of seq and the target freq.
The values shows how the Seg model tends to have less unique sequences for objects which appears frequently as target, while both is not influenced.


- Correlation between *prec_sc* and *seq/sym_cls_rchns* is  0.081 on seg and   -0.262  for both  (0.0413 pval).
The correlation measures how the number of sequences/symbols used for a class is related to the accuracy when target==distractor. We already know that both has problems in discerning objects from the same class, but here we see that these problems are accentuated by the richness of the class itself. That is, more sequences are used for a class worse is the precision when target==distractor.
Moreover we have the opposite for *prec_oc*


- Correlation *prec_oc* and *ambg_rate* is -0.227 in Seg and 0.048 in Both (pval>0.05).
This correlation measures how the times an object appears both as target and distractor influences the precision when having that object as target and another one as distractor. While Both is not bothered by this correlation, Seg has a negative value. This can be explained by saing that seg build representation which are more specific in order to distinguish similar objects but fails when the objects are dissimilar. 

- Correlation *prec_sc* and *ambg_rate* is : -0.044 in Seg and -0.070 in Both  (pval>0.05)..
This is a sanity check, since the more ambiguity there is in a class the more difficult it is to predict it when target==distractor.
On the same wave, we have that the correlation between *ambg_rate* and *accuracy*  (0.0184 pval) is negative for Seg (-0.314) and Both (-0.230) since it is more difficult to predict object, but both finds it slightly easier. 


- Correlation between *cls_comity* and  *prec_oc* is  -0.239 for Seg and  0.035  for Both  (pval>0.05) .  The class comity is a measure of how likely a class is to appear with another one or alone, and its correlation with the precision oc measures how good a model is in differentiating objects of different classes. 
As we derived before, Seg does not perform well in these cases while Both is not bothered.
As a sanity check we have that the correlation between  *cls_comity* and *accuracy*  (0.0153 pval) is negative for both models: in Seg we have a value of  -0.323 while for Both we have a lower value  -0.242.

## Superclass

### Correlation differences

- Correlation between *X_trg!=dstr* and *shared_supercls_seq* (0.0197 pval) is   0.055 for Seg and  0.716 for Both. This relationship measure how the number of shared sequences in a superclass is related to the incorrect predictions when target!=distractor.
Here we see that Both has a high correlation value, which means that the more a superclass shares sequences between its subclasses the more prone the model is to get a prediction wrong.

- Correlation between *freq* and *shared_supercls_seq* (0.0189 pval) is   0.754 for Seg and  0.301 for Both.
We see how Seg tends to share more sequences based on the frequency of the object. Indeed we can look at the *seq_cls_rchns* and see that its correlation with  *shared_supercls_seq*  (0.0043 pval) is  0.843 for Seg and 0.399 for Both. 
Again we see how Seg tends to have more sequences in general than both based on the frequency of the object.

- Correlation between *trg_freq* and *prec_oc* (0.0499 pval) is    -0.667 for Seg and  0.129 for Both.
This correlation highlights the precision in predicting the object when tartget!=distractor in relationship with the target frequency.
We see that Seg finds very difficult to correctly predict this when the object appears frequently as a target, on the other hand Both finds it slightly helpful.

- Correlation between *ambg_rchns_perc* and *prec_oc* (0.0268 pval) is    0.691 for Seg and  -0.022 for Both.
This correlation emphasize the precision in correctly predicting when target!=distractor in relationship to the ratio of unique sequences when target==distractor.
For Seg, more sequences for the t==d means a better differentiation when t!=d.


- Correlation between *cls_comity* and *X_trg==dstr* (0.0252 pval) is    0.683 for Seg and  0.731 for Both.
This correlation highlights how many wrong prediction a model makes when t==d in relationship with the number of distractors t appears with.
We can see that it is slightly higher for Both.



# Answering Marco

## Question one
Marco asks:
"Is there any relationship between the ambiguity rate, the class richness and the target frequency in seg?"

### Metrics Definition
First let's define the metrics:
  - **ambiguity rate** : The ambiguity rate is the number of times the class happens both as a target and a distractors divided the total number of appearances.
The formula is derived as follos:
```(V_trg==dstr+X_trg==dstr)/total```
- **intracls_seq_spcf**  (old class richness): It is defined as the number of unique class sequences divided by the class target frequency. 
- **trg_freq** : The frequency of the current target in the dataset.
- **freq**: number of times the class appears as a target, distractor or other object in the image

### Answer
Following the correlation values between pairs of metrics estimated on a class level:
- Ambiguity rate/ intracls_seq_spcf: 0.185 [Class]
- Ambiguity rate/ trg_freq: -0.161 [Class]
- Ambiguity rate/ freq: 0.155 [Class]
- trg_freq/intracls_seq_spcf: -0.356 [Class]

There is a slight positive correlation between ambiguity rate and intracls_seq_spcf, which means that the more a class appears both as a distractor and a target the more unique sequences it tends to have.

Moreover we can observe a slight negative correlation between the ambiguity rate and the trg_freq while a positive correlation occurs between the ambiguity rate and the frequency. This is a clear factor deriving from the definition of the ambiguity rate which depends on *total=freq* .

Finally there is a considerable negative correlation between the target frequency and the intracls_seq_spcf. We can conclude that more sequences are allocate for classes which have a high ration of appearance with targe==distractor rather than just a high appearance overall.

## Question two
Marco Asks:
"The seg Sender has no idea of the whole context but know only about the target. For this reason the ambiguity_richness metric should be close to 1 "

### Metrics Definitions
- **ambg_rchns** : The ambiguity richness estimates the number of sequences used for a target when target==distractor divided by the number of sequences used when target!=distractor (for the same target).
For example, given the class cat which appears with dog, cat and bike in the dataset, the ambiguity rate is equal to:
```len(Seq(cat,cat))/[len(Seq(cat,dog) +Seq(cat, bike))]``` 
Where `Seq(i,j)` returns the sequences when the target i appears together with the distractor j.
- **cls_comity** : Given a target class t, the class comity is the number of other classes d which appears together with t divided by the total number of classes.
It is a measure of how likely a class is to appear with another one or alone.


### Answer
The mean value of the ambg_rchns across the seg setup is:
- 0.2321 on a class level 

While the mean value of cls_comity is:
 - 0.2749  on a class level.

Consider that there are 70 classes and 12 superclasses, we get the value of the un-normalized cls_comity as:
- 19.24 [Class]

So now we can see how an ambg_rchns of 0.2321 means that 1 class over 19 is taking 23.21% of the sequences, which is 4 times the random baseline (1/19=0.052=5.2%) 


## Question three
Marco Asks:
"What about ambiguity richness in the both setup? "

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

