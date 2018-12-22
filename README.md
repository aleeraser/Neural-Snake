# Neural-Snake

A case study for the application of Deep Neural Networks techniques on a simple game; in this case, the classic Snake game was chosen.

## Implementation

The whole project has been developed in Python 3.7.0 under Mac OS.

The game itself has been developed in two different ways:

- a terminal version, which can also be used headless (i.e. without GUI). This version has been developed for performance purposes; moreover, it can be used in remote ssh sessions (e.g. Amazon AWS, etc.). It has been developed using `curses`.

- a graphic version, developed with `kivy`. This version has been developed for a mere stylist purpose.

The DNN has been developed using `Keras` 2.2.4. The following techniques have been implemented:

- *supervised learning*: the game has been initially trained with (thousand of) random steps, which has been (automatically) labeled and used to train the neural network offline.

- *reinforcement learning*: the network is trained online (in batches) using Q-learning (Deep Q-Learning). The model has been inspired by DeepMind's Atari Deep Q-Learning network.
