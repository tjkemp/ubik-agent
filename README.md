# Ubik-Agent

Status: WIP, actively developed

## About the project

This project started as an excercise in training a Deep Q-Learning agent to navigate in BananaCollector 3D Unity environment.

Currently, I'm working on turning this project into a more general framework to train different agents in several different environments. I'm toying with the idea that this project could work as glue between algorithm libraries, optimization packages and training environments. Although, the purpose is still mainly for myself to learn more about Deep Reinforcement Learning by creating the algorithms.

I don't consider the version to even be 0.1 yet, so expect thing to change and break.

### Implemented algorithms and environments

| Algorithm                | State/Action Spaces   | Example environments |
|--------------------------|-----------------------|----------------------|
| DQN                      | Continuous/Discrete   | BananaCollector (Unity)|
| DDPG                     | Continuous/Continuous | Reacher (Unity)      |
| SarsaMax, Expected Sarsa | Discrete/Discrete     | Taxi (Gym)           |

### Future improvements for version 0.1:
- [x] add DDPG algorithm
- [x] add pytest tests
- [x] add Interaction class to serve as a middleman between the environment and the agent, to remove need for custom handling loops
- [x] add more environments and an easier way to use and install them
- [x] add better measurement (mean/min/max, episode lenghts, details of exploration, etc)
- [x] add hyperparameter tuning and improve the models
- [ ] add Prioritized Experience Replay
- [x] add environments for Sarsa type agents
- [ ] add an example of training and agent from pixel data
- [ ] upgrade ML-Agents package to the latest stable release (mainly because v0.4.0 environments are too unstable to train properly)
- [ ] add PPO, and other algorithms
- [ ] add proper documentation

## BananaCollector Environment

![Trained Agent](https://raw.githubusercontent.com/tjkemp/ubik-agent/assets/images/env/banana.gif)

In the environment agent's task is to collect yellow bananas and avoid purple bananas in a large, square world.

This project would be a simple example on how to train a DQN agent and use it with Unity for anyone interested. The code is extendable for any Unity environment, although currently, the project uses somewhat old versions of dependencies.

### Environment and agent details

The world is a Unity environment and the agent connects to the it through [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) package which is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. The environment is provided by Udacity and it is somewhat similar to single agent version of [BananaCollector](https://github.com/Unity-Technologies/ml-agents/blob/0.4.0/docs/Learning-Environment-Examples.md#banana-collector) environment in Unity's ML-agents package.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a purple banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding purple bananas.

The state space has 37 dimensions and contains the agent's velocity, along with a ray-based perception of objects around the agent's forward direction.  Given this information, the agent has to learn how to best select actions.

Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic and considered solved when the agent gets an average score of +13 over 100 consecutive episodes.

## Other environments

Currently, also an executables to train and run Reacher Unity environment are included (see more information below).

## Installation

Requirements for running the project are Linux, Python version >= 3.6, Unity ML-agents 0.4.0, and Pytorch 1.4.0.

1. Clone the repository

```bash
git clone https://github.com/tjkemp/ubik-agent.git
```

2. Create a python virtual environment and install dependencies

```bash
cd ubik-agent
python -m venv venv
source venv/bin/activate
pip install --no-deps -r requirements.txt
```

3. Install Banana Collector environment for Unity

Download and unzip the environment binary files with the `download.sh` script.

```bash
./environments/download.sh banana
```

Other arguments for the download script are `reacher` and `crawler` for Reacher and Crawler environments binaries respectively.

Note that running the environment requires X Window System. If you are running Windows Subsystem for Linux (WSL), then you need something that provides running X applications such as VcXsrv, Xming, or x410.

## Instructions

All the examples are in the *examples* package directory as classes in their own module files.

For example, the BananaCollector is defined in the module `examples/banana.py`. The module defines a class and methods for training the agent and running the trained agent in the BananaCollector environment.

You can run example modules in the following format: `python -m <package.module> <method> <experiment_name>`

To get help on arguments for each executable, run the module with `-h` switch.

```bash
python -m examples.banana -h
```

### Running an episode with an agent acting randomly

To test the environment with an agent behaving totally randomly run the executable with argument *random*.

```bash
python -m examples.banana random
```

### Training an agent

The python executable takes the directory name of an instance of trained agent as an argument. All the trained agents are saved into the directory *models*.

```bash
python -m examples.banana train my-cool-agent
```

Which will output the following (omitting some messages):

```
Creating an agent (state size of 37, and 4 actions).
Episode 100     Score: 0.00     Best: 11.00     Mean: 2.10
Episode 200     Score: 11.00    Best: 17.00     Mean: 8.30
Episode 300     Score: 19.00    Best: 21.00     Mean: 11.19
Episode 373     Score: 20.00    Best: 25.00     Mean: 13.03
Target score reached in 373 episodes!
```

At the end of the training, the agent model will be saved in the directory *models/my-cool-agent* as `checkpoint.pth`.

### Running an episode with pre-trained agent

```bash
python -m examples.banana run my-cool-agent
```

### Using other included environments

The modules `taxi.py`, `crawler.py` and `reacher.py` work similarly as the BananaCollector environment.

## Licenses and acknowledgements

- This project is licensed under [MIT LICENSE](LICENSE).
- The original code was inspired ny [Udacity Deep Reinforcement Learning nanodegree](https://github.com/udacity/deep-reinforcement-learning/) materials.

## Author

- [tjkemp](https://github.com/tjkemp)
