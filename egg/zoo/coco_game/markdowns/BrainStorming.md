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