# Ubik-Agent

## About the project

This project trains a Deep Q-Learning agent in Python to navigate and collect bananas in a large, square world.

![Trained Agent](files/banana-environment.gif)

This project would be a simple example on how to train a DQN agent and use it with Unity for anyone interested. The code is extendable for any Unity environment, although currently, the project uses somewhat old versions of dependencies.

Requirements for running the project are Linux, Python version >= 3.6, Unity ML-agents 0.4.0, and Pytorch 0.4.

## Unity Environment and agent details

The world is a Unity environment and the agent connects to the it through [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) package which is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. The environment is provided by Udacity and it is somewhat similar to single agent version of [BananaCollector](https://github.com/Unity-Technologies/ml-agents/blob/0.4.0/docs/Learning-Environment-Examples.md#banana-collector) environment in Unity's ML-agents package.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with a ray-based perception of objects around the agent's forward direction.  Given this information, the agent has to learn how to best select actions.

Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic and considered solved when the agent gets an average score of +13 over 100 consecutive episodes.

## Installation

1. Clone the repository

```bash
git clone `repository url`
```

2. Create a python virtual environment and install dependencies

```bash
cd `repository directory name`
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Install Banana Collector environment for Unity

If you want to see the trained agent in the environment on your screen, download and unzip the prepared environment binary files.

```bash
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip
```

Note that running the simulation requires X Window System. If you are running Windows Subsystem for Linux (WSL), then you need something that provides running X applications such as VcXsrv, Xming, or x410.

Or, if you instead, want to run the environment without visualization, get the headless version of the environment. You will not be able to watch the agent, but you will be able to train the agent.

```bash
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
unzip Banana_Linux_NoVis.zip
```

## Instructions

The project has two python executables, `train.py` and `play.py`, for training an agent and simulating one episode.

To get help on arguments for each executable, run them with `-h` switch, for example:

```bash
python play.py -h
```

### Running an episode with pre-trained agent

First, enter the python virtual environment if it's not activated already.

```bash
python play.py --unity_env files/checkpoint.pth
```

### Training an agent

First, enter the python virtual environment if it's not activated already.

```bash
python train.py --target_score 13
```

Which will output the following (omitting some Unity's messages):

```
Creating an agent (state size of 37, and 4 actions).
Episode 100     Average Score: 0.892    Epsilon: 0.61
Episode 200     Average Score: 3.79     Epsilon: 0.37
Episode 300     Average Score: 7.24     Epsilon: 0.22
Episode 400     Average Score: 10.41    Epsilon: 0.13
Episode 500     Average Score: 12.39    Epsilon: 0.08
Episode 533     Average Score: 13.08    Epsilon: 0.07
Target score reached in 533 episodes!   Average Score: 13.08
```

At the end of the training, the agent model will be saved in the current directory as `checkpoint.pth`.

## Licenses and acknowledgements

- The code is heavily based on [Udacity Deep Reinforcement Learning nanodegree](https://github.com/udacity/deep-reinforcement-learning/) code and materials and is thus continued to be licensed under [MIT LICENSE](LICENSE).

## Author

- [tjkemp](https://github.com/tjkemp)
