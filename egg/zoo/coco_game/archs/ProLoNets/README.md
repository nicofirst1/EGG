# ProLoNets
Public implementation of "Encoding Human Domain Knowledge to Warm Start Reinforcement Learning" from AAAI'21

### Requirements

I've gone ahead and made two separate virtualenvs for the OpenAI gym environments and the StarCraft II environments, both of which are built on Python 3.6. In order to work with the SC2 environments, you must have Python >= 3.6, and then installing the requirements in the `sc2_requirements.txt` file should do it. For the gym environments, any Python that works with OpenAI Gym _should_ work, but I haven't tested this.

### Running Experiments

All of the code to run various domains lives in the `runfiles/` directory. 
All file involve a few command line arguments, which I'll review now:

* `-a` or `--agent_type`: Which agent should play through the domain. Details below. Default: `prolo`
* `-e` or `--episodes`: How many episodes to run for. Default: 1000
* `-s` or `--sl_init`: Should the agent be trained via imitation learning first? Only applies if `agent_type` is `fc`.Default: False
* `-rand`: Should the ProLoNet agent be randomly-initialized? Include flag to set to `True`.
* `-deep`: Should the ProLoNet include dynamic growth? Include flag to set to `True`.
* `-adv`: Should the ProLoNet be an "M-Mistake" agent? Include the flag to set to `True`. The probability itself is hard-coded in the ProLoNet file at line 35.

For the `-a` or `--agent_type` flag, valid options are:
* `prolo` for a normal ProLoNet agent
* `random` for random actions (not available in full game of StarCraftII)
* `heuristic` for the heuristic only (not available in the full game of StarCraftII)
* `fc` for a fully-connected agent
* `lstm` for an LSTM agent
* `djinn` for a DJINN agent

#### gym_runner.py

This file runs both of the OpenAI gym domains from the paper, namely cart pole and lunar lander. It has one additional command line argument:
* `-env` or `--env_type`: Which environment to run. Valid options are `cart` and `lunar`. Default: `cart`
In order to run, you  must have the a Python environment with the OpenAI Gym installed. Furthermore, you  must have box2d-py if you want the lunar lander agents to run. The `gym_requirements.txt` file should have everything necessary for a Python 3.6 environment.

Running a ProLoNet agent on lunar lander for 1500 episodes looks like:
```
python gym_runner.py -a prolo -deep -e 1500 -env lunar
```
For the _LOKI_ agent:
```
python gym_runner.py -a fc -e 1500 -env lunar -s
```

#### minigame_runner.py

This file runs the FindAndDefeatZerglings minigame from the SC2LE. Running this is exactly the same as the `gym_runner.py` runfile, with the exception that no `--env_type` flag exists for this domain. You must also have all of the StarCraft II setup complete, which means having a valid copy of StarCraft II, having Python >= 3.6, and installing the requirements from the `sc2_requirements.txt` file. For information on setting up StarCraft II, refer to [Blizzard's Documentation](https://github.com/Blizzard/s2client-proto) and for the minigame itself, you'll need the map from [DeepMind's repo](https://github.com/deepmind/pysc2).

Running a ProLoNet agent:
```
python minigame_runner.py -a prolo -deep -e 1000
```
And a fully-connected agent:
```
python minigame_runner.py -a fc -e 1000
```
And an LSTM agent:
```
python minigame_runner.py -a lstm -e 1000
```
And a DJINN agent:
```
python minigame_runner.py -a djinn -e 1000
```

#### sc_runner.py

This file runs the full SC2 game against in-game AI. In game AI difficulty is set on lines 836-838. Simply changing "Difficult.VeryEasy" to "Difficulty.Easy", "Difficulty.Medium", or "Difficulty.Hard" does the trick. Again, you'll need SC2 and all of the requirements for the appropriate Python environment, as discussed above.
Running a ProLoNet agent:
```
python sc_runner.py -a prolo -e 500
```
And a random ProLoNet agent:
```
python sc_runner.py -a prolo -rand -e 500
```

#### Citation
If you use this project, please cite our work! Our arXiv paper is available here: https://arxiv.org/abs/1902.06007 and a newer draft will soon be published in AAAI'21!