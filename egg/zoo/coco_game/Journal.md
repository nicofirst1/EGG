# Experiments
These experiments where run with 15 classes from the coco dataset (skipped first 5)
The performance was measured by the test class accuracy.
Whenever the `***` is present then more experiments should be done.

## Hidden size
In this part the variation of the parameters `sender/receiver_hidden_size` will be studied.
This parameter controls the two different things:

    1. In the "RnnReceiverDeterministic" is the hidden dimension of the `RnnEncoder` which translates to the dimensions of the chosen rnn cells
    2. In the "RnnSenderReinforce" the dimension controls the hidden to output linear module

Both are directly tied to the deepness of the language interpretation part.

The parameters were changed together from a min of 16 to a max of 256.
Moreover, two runs were executed per each configuration in order to get a better generalization, the variance observed between runs with the same configuration is in range [2,12]%

Overall the best results where given by a value of 128. The trends seems to follow a linear fashion where the increase in the hidden number maps to an increased performance. 
This is not true for the 256 value which yielded worst results than 64. 

Optimal values: `sender_hidden_size:128` `receiver_hidden_size:128`

## Vocab
The vocabulary size and length (`vocab_size, max_len`) control the message structure and are the most influential parameters so far.
Again there seems to be a linear relationship between the `vocab_size` and the accuracy which increases steadily until a value of 30 (15 classes).
For greater values (>50) the performances see a drastic drop (30%).
On the other hand the `max_len` seems to be less influential as long as the product of its value and the `vocab_size` is kept under 40 with `vocab_size>10`.
Practically speaking the language variance defined by the `vocab_size` parameter seems to be the main factor contributing to a good performance which can deal with short message sequences.
On the other hand the agents appear to be confused when the `max_len` increases too much.

Optimal values: `vocab_size:20` `max_len:3`

## Optimizer
As for now there are 3 available optimizers:
    1. Adam
    2. SGD
    3. AdaGrad

The results show that adam is in general better since it starts from a high value (+40% than others) but the learning stop shortly after gaining +10 %
On the other side, a much longer training was performed with the SGD which yielded similar results to adam in x25 epochs.

Optimal value : adam

## Cell Types ***
In this part the parameters `sender/receiver_cell_type` were varied independently.

## Loss
In here the parameters `cross/kl_lambda` and `use_class_weights` where modified. 
It is clear that kl loss cannot compensate for the cross entropy, indeed when `cross_lambda=0` no values for the other two parameters can compensate a loss in 40% performance.
On the other hand `kl_lambda` seems to help slightly (3%) when present with a linear contribute.
*** use_class_weights

Optimal value : `cross_lambda=1` `kl_lambda=0.3` 

## Number of Layers
In this part the parameters `sender/receiver_num_layers` were tested.
These parameters represent the number of recursions a rnn cell (independently of the type) has.

From the experiments it is clear that any value >1 comes with a drop in performance. 
More tests should be performed with more epochs to see if the performance can increase ***,