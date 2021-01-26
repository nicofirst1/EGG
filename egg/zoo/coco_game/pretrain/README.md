This dir contains the code necessary to pretrain the sender on the classification task. The sender weights can then be
imported on the actual training.

- [game](game.py): game module to train sender
- [pretrain](pretrain.py): main python script to be called
- [bash pretrain](bash_pretrain.sh): used to call the previous on bash with the [default args](../args.sh)