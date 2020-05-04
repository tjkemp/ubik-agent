# Ubik-Agent

## About the project

This project started as an excercise in training a Deep Q-Learning agent to navigate in BananaCollector 3D Unity environment.

Currently, I'm working on turning this project into a more general library to train different agents in several different Unity environments. The purpose is to myself learn more about Deep Reinforcement Learning by creating the algorithms.

### Future improvements:
- [x] add DDPG algorithm
- [x] add pytest tests
- [x] add Interaction class to serve as a middleman between the environment and the agent, to remove need for custom handling loops
- [ ] add more environments and an easier way to use and install them
- [ ] add better measurement (mean/min/max, episode lenghts, details of exploration, etc)
- [ ] add Prioritized Experience Replay
- [ ] add hyperparameter tuning and improve the models
- [ ] add capability to train the agent from pixel data
- [ ] upgrade ML-Agents package to the latest Release 1
- [ ] add PPO, Dueling DQN and other algorithms (to be decided)

## BananaCollector Environment

![Trained Agent](files/banana-environment.gif)

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

If you want to see the trained agent in the environment on your screen, download and unzip the prepared environment binary files.

```bash
cd environments
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip
```

Note that running the simulation requires X Window System. If you are running Windows Subsystem for Linux (WSL), then you need something that provides running X applications such as VcXsrv, Xming, or x410.

## Instructions

The project has single python executable: `banana.py`, which is used both for training an agent and running the trained agent.

To get help on arguments for each executable, run them with `-h` switch, for example:

```bash
python banana.py -h
```

### Running an episode with pre-trained agent

First, enter the python virtual environment if it's not activated already.

The python executable takes the directory name of the trained agent as an argument. All the trained agents are in directory *models*.

```bash
python banana.py run banana
```

### Training an agent

First, enter the python virtual environment if it's not activated already.

The python executable takes the directory name of the trained agent as an argument. All the trained agents are in directory *models*.

```bash
python banana.py train my-cool-agent
```

Which will output the following (omitting some Unity's messages):

```
Creating an agent (state size of 37, and 4 actions).
Episode 100     Score: 0.00     Best: 11.00     Mean: 2.10
Episode 200     Score: 11.00    Best: 17.00     Mean: 8.30
Episode 300     Score: 19.00    Best: 21.00     Mean: 11.19
Episode 373     Score: 20.00    Best: 25.00     Mean: 13.03
Target score reached in 373 episodes!
```

At the end of the training, the agent model will be saved in the *models* directory as `checkpoint.pth`.

## Licenses and acknowledgements

- The original code is based on [Udacity Deep Reinforcement Learning nanodegree](https://github.com/udacity/deep-reinforcement-learning/) materials and is thus continued to be licensed under [MIT LICENSE](LICENSE).

## Author

- [tjkemp](https://github.com/tjkemp)
