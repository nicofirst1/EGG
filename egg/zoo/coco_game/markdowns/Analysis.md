# Class vs Superclass

The per class analysis focuses more on the accuracy of each prediction, in this case all the classes are seen as diverse objects that needs to be predicted. 

On the other hand, the superclass analysis focuses more on a set of objects sharing similar characteristics. In this case it emphasize the difficulty in predicting classes with the same superclass.

TD;DR
Xclass analysis is more about accuracy, Xsuperclass is more about generalization and differentiation.

# Language

Seg has more sequences than both (1360 vs 1135 ) normalized by the max combinations of sequences 531630 (0.21% vs 0.26%).
Moreover both has a  lower average message length (5.92 vs 6), due to the occurrence of sequences with length 1 while seg sticks to the max sequence length available.

Seg has higher values for all types of specificity, which means that it has more sequences which are used once for class/superclass. Instead both has a lower number of sequences overall which are shared more across objects.

## Sequence specificity

Another interesting aspect of the data is seq_spec.
The sequence specificity is the proportion of sequences used mainly (more than 99% of the times) for one superclass divided by the number of sequences.
Example:
Having a sequence specificity of 10% means that 10% of all the sequences appear only with one class.

In the Both model we have that 28.4% of the sequences are unique per superclass while in Seg is 30.9%. This means that Seg uses more sequences to better differentiate between classes while Both shares them.

### Decreasing trend
By increasing the number of epochs, thus increasing the combination of (target, distractor) seen by the model, we have a decreasing trend for the sequence specificity reported in the table below

| Seed | Training Epochs | Accessories Num | Seq_Spec Seg | shared_supercls_seq Seg | Seq_Spec Both | shared_supercls_seq Both |
|------|-----------------|-----------------|--------------|------------------|---------------|-------------------|
| Mean | 1               | 73.6            | 56.92        | 2                | 51.68         | 2.12              |
| Mean | 2               | 148             | 48.94        | 3.36             | 44.16         | 2.827             |
| Mean | 3               | 222.8           | 44.14        | 4.38             | 40.9          | 4.77              |
| Mean | 5               | 388.4           | 37.82        | 6.71             | 36            | 6.7               |
| 31   | 10              | 747             | 31.9         | 7.03             | 28.9          | 9.78              |
| 32   | 10              | 740             | 29.5         | 9.94             | 28.1          | 10.19             |
|      |                 |                 |              |                  |               |                   |

On the other hand the shared_supercls_seq increases linearly with the dataset dimension. This is a common thread for both Seg and Both, which means that the model is guided to increase the sequence sharing when the dataset is various. 




# Class

## Correlations


### General 


- Ambiguity rate is high pos (0.96) with amb richness: classes with more target==distr have more sequences.

- Ambiguity rate is neg (-0.314) with accuracy: ambiguous objects are more difficult to predict

-  Class richness is high pos (0.88) to the class frequency: more frequent classes have more symbols/sequences

-  Target freq is high neg (-0.53) to frequency: classes which are more frequent appear less as a target and more as other classes 


### Differences

- Correlation between *ambiguity_rate* and *accuracy*  (0.0184 pval) is  -0.314/-0.230  . This means that Both  finds less difficult to predict  a class when the object is ambiguous.
-  Correlation between *X_trg==dstr* and *othr_clss_num*  (0.043 p) is 0.043/0.266. The correlation measure how difficult is the prediction in relation with the number of classes present in the overall image. Since the Seg model does not have access to the image the correlation is practically zero, while for the both model is positive. This means that both finds it more difficult to distinguish target from distractor (when they are the same class) when there is a high number of objects in the image.
-  Correlation between *othr_clss_num* and *prec_sc* (0.0176 p) is 0.077/-0.303.  The correlation measures the precision in correctly predicting a class when target==distractor in relationship with the number of objects in the image. Here too we see how the both model is confused by number of object in the image while the seg one nope.
-  Correlation between *trg_freq* and *intracls_seq_spcf*   (0.0061 pval) is -0.356/0.015. The correlation measures how many unique senquencies a classes has in relationship with its target frequency. Since the *intracls_seq_spcf* is multiplied by the *trg_freq* in its computation we actually have a relationship between the number of seq and the target freq.
The values shows how the Seg model tends to have less unique sequences for objects which appears frequently as target, while both is not influenced.
- Correlation between *prec_sc* and *seq/sym_cls_rchns* is  0.081/-0.262. The correlation measures how the number of sequences/symbols used for a class is related to the accuracy when target==distractor. We already know that both has problems in discerning objects from the same class, but here we see that these problems are accentuated by the richness of the class itself. That is, more sequences are used for a class worse is the precision when target==distractor.


#### P> 0.5



- On the other hand, correlation between *othr_clss_num* and *prec_oc* is   -0.242/0.024  .
The correlation measures the precision in correctly predicting a class when target!=distractor in relationship with the number of objects in the image. 
In a completely different results as before, here we see how the number of objects in the image influences negatively Seg when it needs to differentiate between to objects coming from different classes.


- Correlation between *freq* and the two precisions *prec_oc*   and *prec_sc* (0.0194 p) confirms what previously said. Indeed for the Seg model this correlation is negative with *prec_oc* (-0.251) and zero with *prec_sc* (0.049), on the other hand, Both has a negative correlation with *prec_sc* (-0.299) and zer with *prec_oc* (0.041).
This means that Seg is fooled in predicting targets which appears frequently when the distractor is not the same class, on the other hand both is fooled when the opposite is true.



- Correlation *prec_oc* and *ambg_rate* is -0.227 in Seg and 0.048 in Both (pval>0.05).
This correlation measures how the times an object appears both as target and distractor influences the precision when having that object as target and another one as distractor. While Both is not bothered by this correlation, Seg has a negative value. This can be explained by saing that seg build representation which are more specific in order to distinguish similar objects but fails when the objects are dissimilar. 

- Correlation *prec_sc* and *ambg_rate* is : -0.044 in Seg and -0.070 in Both  (pval>0.05)..
This is a sanity check, since the more ambiguity there is in a class the more difficult it is to predict it when target==distractor.
On the same wave, we have that the correlation between *ambg_rate* and *accuracy*  (0.0184 pval) is negative for Seg (-0.314) and Both (-0.230) since it is more difficult to predict object, but both finds it slightly easier. 


- Correlation between *cls_comity* and  *prec_oc* is  -0.239 for Seg and  0.035  for Both  (pval>0.05) .  The class comity is a measure of how likely a class is to appear with another one or alone, and its correlation with the precision oc measures how good a model is in differentiating objects of different classes. 
As we derived before, Seg does not perform well in these cases while Both is not bothered.
As a sanity check we have that the correlation between  *cls_comity* and *accuracy*  (0.0153 pval) is negative for both models: in Seg we have a value of  -0.323 while for Both we have a lower value  -0.242.

# Super Class

## Correlation 

### Differences 

- Correlation between *X_trg!=dstr* and *shared_supercls_seq* (0.0197 pval) is   0.055/0.716.  This relationship measure how the number of shared sequences in a superclass is related to the incorrect predictions when target!=distractor.
Here we see that Both has a high correlation value, which means that the more a superclass shares sequences between its subclasses the more prone the model is to get a prediction wrong.
-  Correlation between *freq* and *shared_supercls_seq* (0.0189 pval) is  0.754/0.301 . We see how Seg tends to share more sequences based on the frequency of the object. Indeed we can look at the *seq_cls_rchns* and see that its correlation with  *shared_supercls_seq*  (0.0043 pval) is  0.843 for Seg and 0.399 for Both. 
Again we see how Seg tends to have more sequences in general than both based on the frequency of the object.
- Correlation between *trg_freq* and *prec_oc* (0.0499 pval) is  -0.667/0.129. This correlation highlights the precision in predicting the object when tartget!=distractor in relationship with the target frequency.
We see that Seg finds very difficult to correctly predict this when the object appears frequently as a target, on the other hand Both finds it slightly helpful.
- Correlation between *ambg_rchns_perc* and *prec_oc* (0.0268 pval)  is 0.691/-0.022.
This correlation emphasize the precision in correctly predicting when target!=distractor in relationship to the ratio of unique sequences when target==distractor.
For Seg, more sequences for the t==d means a better differentiation when t!=d.
- Correlation between *cls_comity* and *X_trg==dstr* (0.0252 pval) is  0.683/0.731.
This correlation highlights how many wrong prediction a model makes when t==d in relationship with the number of distractors t appears with. We can see that it is slightly higher for Both.




# Tl;DR

The seg model is better in predicting the actual target but it does so at the cost of higher sequences specialization
and number. On the other hand the both model is slightly worse at predicting the target but it share sequences across
superclasses.

Seg is better at differentiating between objects with the same class while Both is better when target!=distractor.




###############################################################################################




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


