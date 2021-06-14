import sys

import torch
from torch import nn
from torch.distributions import Categorical

sys.path.insert(0, '../')
from opt_helpers import replay_buffer
from agents.vectorized_prolonet_helpers import swap_in_node, add_level, \
    init_general_nets
import copy


class DeepProLoNet(nn.Module):
    def __init__(self,
                 bot_name='ProLoNet',
                 input_dim=4,
                 output_dim=2,
                 use_gpu=False,
                 vectorized=False,
                 randomized=False,
                 adversarial=False,
                 deepen=True,
                 epsilon=0.9,
                 epsilon_decay=0.95,
                 epsilon_min=0.05,
                 deterministic=False,
                 vocab_size=4,
                 max_len=2,
                 image_processor=None,
                 ):

        super(DeepProLoNet, self).__init__()

        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.bot_name = bot_name
        self.use_gpu = use_gpu
        self.vectorized = vectorized
        self.randomized = randomized
        self.adversarial = adversarial
        self.deepen = deepen
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.adv_prob = .05
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.deterministic = deterministic
        self.image_processor = image_processor

        self.vocab_size = vocab_size
        self.max_len = max_len - 1

        if vectorized:
            self.bot_name += '_vect'
        if randomized:
            self.bot_name += '_rand'
        if use_gpu:
            self.bot_name += '_gpu'
        if deepen:
            self.bot_name += '_deepening'

        self.action_network, self.value_network = init_general_nets(input_dim=input_dim, output_dim=self.vocab_size,
                                                                    use_gpu=use_gpu, vectorized=vectorized)

        self.deeper_action_network = [add_level(self.action_network, use_gpu=use_gpu) for _ in range(self.max_len)]

        self.deeper_actor_opt = [torch.optim.RMSprop(self.deeper_action_network[i].parameters()) for i in
                                 range(self.max_len)]

        self.num_times_deepened = 0
        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = None
        self.last_deep_value_pred = [None] * output_dim
        self.full_probs = None
        self.deeper_full_probs = None
        self.reward_history = []
        self.num_steps = 0

    def forward(self, observation):

        returns = []
        observation = self.image_processor(observation)

        if observation.ndim > 1:
            batch_size = observation.shape[0]
            for i in range(batch_size):
                returns.append(self.get_message(observation[i]))

            sequence = torch.stack([returns[i][0] for i in range(batch_size)])
            logits = torch.stack([returns[i][1] for i in range(batch_size)])
            entropy = torch.stack([returns[i][2] for i in range(batch_size)])

            if self.use_gpu:
                sequence = sequence.cuda()
                entropy = entropy.cuda()
                logits = logits.cuda()

            return sequence, logits, entropy
        else:
            return self.get_message(observation)

    def get_message(self, observation):

        sequence = []
        logits = []
        entropy = []

        obs = torch.Tensor(observation.cpu())
        obs = obs.view(1, -1)
        self.last_state = obs
        if self.use_gpu:
            obs = obs.cuda()
        probs = self.action_network(obs)
        probs = probs.view(-1).cpu()
        self.full_probs = probs

        m = Categorical(probs)
        entropy.append(m.entropy())

        if self.training:
            action = m.sample()
        else:
            action = torch.argmax(probs)

        log_probs = m.log_prob(action)
        logits.append(log_probs)
        sequence.append(action)

        for step in range(self.max_len):
            deeper_probs = self.deeper_action_network[step](obs)
            deeper_probs = deeper_probs.view(-1).cpu()
            deep_m = Categorical(deeper_probs)
            entropy.append(deep_m.entropy())
            if self.training:
                action = deep_m.sample()
            else:
                action = torch.argmax(deeper_probs)
            deep_log_probs = deep_m.log_prob(action)
            logits.append(deep_log_probs)
            sequence.append(action)

        sequence.append(0)
        logits.append(0)
        entropy.append(0)

        sequence = torch.as_tensor(sequence)
        logits = torch.as_tensor(logits)
        entropy = torch.as_tensor(entropy)

        return sequence, logits, entropy

    def end_episode(self, timesteps, num_processes):
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self, go_deeper=self.deepen)
        self.num_steps += 1
        # Copy over new decision node params from shallower network to deeper network
        bot_name = '../txts/' + self.bot_name + str(num_processes) + '_processes'
        with open(bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def lower_lr(self):
        for param_group in self.ppo.actor_opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
        for param_group in self.ppo.critic_opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

    def deepen_networks(self):
        if not self.deepen or self.num_times_deepened > 8:
            return
        self.entropy_leaf_checks()
        # Copy over shallow params to deeper network
        for weight_index in range(len(self.action_network.layers)):
            new_act_weight = torch.Tensor(self.action_network.layers[weight_index].cpu().data.numpy())
            new_act_comp = torch.Tensor(self.action_network.comparators[weight_index].cpu().data.numpy())

            if self.use_gpu:
                new_act_weight = new_act_weight.cuda()
                new_act_comp = new_act_comp.cuda()

            self.deeper_action_network.layers[weight_index].data = new_act_weight
            self.deeper_action_network.comparators[weight_index].data = new_act_comp
        for weight_index in range(len(self.value_network.layers)):
            new_val_weight = torch.Tensor(self.value_network.layers[weight_index].cpu().data.numpy())
            new_val_comp = torch.Tensor(self.value_network.comparators[weight_index].cpu().data.numpy())
            if self.use_gpu:
                new_val_weight = new_val_weight.cuda()
                new_val_comp = new_val_comp.cuda()
            self.deeper_value_network.layers[weight_index].data = new_val_weight
            self.deeper_value_network.comparators[weight_index].data = new_val_comp

    def entropy_leaf_checks(self):
        leaf_max = torch.nn.Softmax(dim=0)
        new_action_network = copy.deepcopy(self.action_network)
        changes_made = []
        for leaf_index in range(len(self.action_network.action_probs)):
            existing_leaf = leaf_max(self.action_network.action_probs[leaf_index])
            new_leaf_1 = leaf_max(self.deeper_action_network.action_probs[2 * leaf_index + 1])
            new_leaf_2 = leaf_max(self.deeper_action_network.action_probs[2 * leaf_index])
            existing_entropy = Categorical(existing_leaf).entropy().item()
            new_entropy = Categorical(new_leaf_1).entropy().item() + \
                          Categorical(new_leaf_2).entropy().item()

            if new_entropy + 0.1 <= existing_entropy:
                with open('../txts/' + self.bot_name + '_entropy_splits.txt', 'a') as myfile:
                    myfile.write('Split at ' + str(self.num_steps) + ' steps' + ': \n')
                    myfile.write('Leaf: ' + str(leaf_index) + '\n')
                    myfile.write('Prior Probs: ' + str(self.action_network.action_probs[leaf_index]) + '\n')
                    myfile.write('New Probs 1: ' + str(self.deeper_action_network.action_probs[leaf_index * 2]) + '\n')
                    myfile.write(
                        'New Probs 2: ' + str(self.deeper_action_network.action_probs[leaf_index * 2 + 1]) + '\n')

                new_action_network = swap_in_node(new_action_network, self.deeper_action_network, leaf_index,
                                                  use_gpu=self.use_gpu)
                changes_made.append(leaf_index)
        if len(changes_made) > 0:
            self.num_times_deepened += 1
            self.action_network = new_action_network

            if self.action_network.input_dim > 100:
                new_actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=1e-4)
            elif self.action_network.input_dim >= 8:
                new_actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=1e-3)
            else:
                new_actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=1e-3)

            self.ppo.actor = self.action_network
            self.ppo.actor_opt = new_actor_opt

            for change in changes_made[::-1]:
                self.deeper_action_network = swap_in_node(self.deeper_action_network, None, change * 2 + 1,
                                                          use_gpu=self.use_gpu)
                self.deeper_action_network = swap_in_node(self.deeper_action_network, None, change * 2,
                                                          use_gpu=self.use_gpu)

            self.deeper_actor_opt = torch.optim.RMSprop(self.deeper_action_network.parameters(), lr=1e-2)

    def __getstate__(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'deeper_action_network': self.deeper_action_network,
            'deeper_value_network': self.deeper_value_network,
            'actor_opt': self.actor_opt,
            'value_opt': self.value_opt,
            'deeper_actor_opt': self.deeper_actor_opt,
            'deeper_value_opt': self.deeper_value_opt,
            'bot_name': self.bot_name,
            'use_gpu': self.use_gpu,
            'vectorized': self.vectorized,
            'randomized': self.randomized,
            'adversarial': self.adversarial,
            'deepen': self.deepen,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'num_times_deepened': self.num_times_deepened,
            'deterministic': self.deterministic,
        }

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

    def duplicate(self):
        new_agent = DeepProLoNet(distribution='one_hot',
                                 bot_name=self.bot_name,
                                 input_dim=self.input_dim,
                                 output_dim=self.output_dim,
                                 use_gpu=self.use_gpu,
                                 vectorized=self.vectorized,
                                 randomized=self.randomized,
                                 adversarial=self.adversarial,
                                 deepen=self.deepen,
                                 epsilon=self.epsilon,
                                 epsilon_decay=self.epsilon_decay,
                                 epsilon_min=self.epsilon_min,
                                 deterministic=self.deterministic
                                 )
        new_agent.__setstate__(self.__getstate__())
        return new_agent
