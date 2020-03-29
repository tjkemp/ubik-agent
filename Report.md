# Project report

## Learning Algorithm

The implemented learning algorithm to solve the BananaBrain Unity environment is Deep
Q-Learning algorithm first introduced by DeepMind's 
[research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

Q-Learning is a model-free reinforcement learning algorithm to learn a policy for an agent. 
In Deep Q-Learning, a neural network represents the Q. More specifically, the DQN takes the
state as an input and returns the corresponding predicted action values for each possible
actions.

## Plot of Rewards

![Score plot](files/scores.png?raw=true "Plot of rewards")

Plot of sum rewards collected while training for 1800 episodes with following hyperparameters:
- gamma 0.99, starting epsilon 1.0, minimum epsilon 0.01 and epsilon decay of 0.995,
- neural net learning_rate 5e-4, batch_size 64, and
- replay buffer size 1e5 update interval 4

## Ideas for Future Work

- Hyperparameter tuning and neural network architecture improvements could make the agent learn faster.
- Immediate step before any of the following ideas would be to upgrade the code use the latest stable versions of Pytorch and Unity's ml-agents package. The packages are more than two years old.
- Good following step would be not to use the headless environment, but instead, visualize the agent's behavior in the environment.
- Extending the agent to use CNNs to read raw pixel values would be an exciting future improvement.
- Finally, extensions such as Double DQN, Prioritized experience replay, or Dueling DQN could be implemented improve the agent's performance even further.
