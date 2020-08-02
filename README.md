# Ubik-Agent

Status: WIP, actively developed

## About the project

This project started as an excercise in training a Deep Q-Learning agent to navigate in BananaCollector (Unity ML-Agents 0.4.0) environment.

Currently, I'm working on turning this project into a more general framework to train RL agents in any OpenAI Gym environment with any of the most common RL Agent libraries. I think, too often environments, agents and the code between (providing visualization, optimization, persistance, reproducibility) are tightly coupled. Although this framework could work as glue between algorithm libraries and training environments, the main purpose is still mainly for myself to learn more about Deep Reinforcement Learning.

I don't consider the version to be v0.1 quite yet, so expect thing to change and break.

### Implemented algorithms and environments

| Algorithm                | State/Action Spaces   | Example environments |
|--------------------------|-----------------------|----------------------|
| DQN                      | Continuous/Discrete   | ~~BananaCollector (Unity)~~|
| DDPG                     | Continuous/Continuous | ~~Reacher (Unity)~~      |
| Q-learning | Discrete/Discrete     | Taxi (Gym)           |
| Expected Sarsa | Discrete/Discrete     | Taxi (Gym)           |

### Improvements for version 0.1:
- [x] add DDPG algorithm
- [x] add pytest tests
- [x] add Interaction class to serve as a middleman between the environment and the agent, to remove need for custom handling loops
- [x] add more environments and an easier way to use and install them
- [x] add better measurement (mean/min/max, episode lenghts, details of exploration, etc)
- [x] add hyperparameter tuning example
- [x] add Prioritized Experience Replay
- [x] add Sarsa/Q-Learning agents and examples
- [x] add callbacks for flexible logging, and adapters to handle non-standard environments and agents
- [x] make runs reproducible
- [x] remove legacy ML-Agents v0.4.0 examples
- [ ] add OpenAI Gym examples
- [ ] update usage examples and architecture diagrams

### Improvements for version 0.2
- [ ] add as simple as can be vanilla versions of DQN and DDPG
- [ ] add basic algorithms like Policy Iteration and Vanilla Policy Gradient
- [ ] make DQN and DDPG agents capable of learning from pixel data
- [ ] add ML-Agents examples

### Improvements for later version
- [ ] add proper documentation
- [ ] add examples of using algorithms from other packages
- [ ] add more advanced variations of DQN
- [ ] add more advanced algorithms like PPO and SAC


## Installation

Requirements for running the project are Linux (or similar, like WSL), Python version >= 3.6 and Pytorch 1.4.0.

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

## Instructions

All the examples are in the *examples* package directory as classes in their own files. (Currently only Gym Taxi v3.)

You can run example modules in the following format: `python -m <package.module> <method> <experiment_name>`

To get help on arguments for each executable, run the module with `-h` switch.

```bash
$ python -m examples.banana -h

usage: taxi.py [-h] {optimize,random,run,train} ...

optional arguments:
  -h, --help            show this help message and exit

method:
  {optimize,random,run,train}
                        a method in the class
    optimize            optimize
    random              random
    run                 run
    train               train
```

### Running an episode with an agent acting randomly

To test the environment with an agent behaving totally randomly run the executable with argument *random*.

```bash
python -m examples.taxi random
```

### Training an agent

The python executable takes the directory name of an instance of trained agent as an argument. All the trained agents are saved into the directory *models*.

```bash
python -m examples.taxi train my-cool-agent
```

At the end of the training, the agent model will be saved in the directory *models/my-cool-agent* as `checkpoint.pth`.

### Running an episode with pre-trained agent

```bash
python -m examples.taxi run my-cool-agent
```

## Licenses and acknowledgements

- This project is licensed under [MIT LICENSE](LICENSE).
- The original code was inspired by [Udacity Deep Reinforcement Learning nanodegree](https://github.com/udacity/deep-reinforcement-learning/) materials.

## Author

- [tjkemp](https://github.com/tjkemp)
