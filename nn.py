#!/usr/bin/env python3

# Inspired from https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3

"""
    Snake looks in 8 direction. For each direction, it sees:
    - distance between head and food (if any)
    - distance between head and part of body (if any)

    Input: 2x8
    Output: 4x1 (direction)

    The idea is to minimize the distance between the snake head and the foot without dying.
"""

import random
import sys
from collections import Counter
from random import randint
from statistics import mean
from threading import Thread
from time import sleep
from traceback import print_exc

import numpy as np
from keras.layers import Activation, Dense, InputLayer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical

from snake_game_kivy import Direction, Snake

TRAIN_CLOCK = 0.004
TEST_CLOCK = 0.1


class SnakeNN:
    def __init__(self, train_games=100, test_games=100, goal_steps=100, lr=1e-2, filename='snake_nn.keras', game=None):
        self.initial_games = train_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vector_direction_map = [
            [[-1, 0], Direction.LEFT],
            [[0, 1], Direction.UP],
            [[1, 0], Direction.RIGHT],
            [[0, -1], Direction.DOWN]
        ]

        self.game = game

    def generate_training_data(self):
        training_data = []
        for _ in range(self.initial_games):

            prev_observation = self.game.is_neighborhood_blocked()
            for _ in range(self.goal_steps):
                # action: -1 (turn left), 0 (straight), 1 (turn right)
                # game action: Direction.[LEFT, UP, RIGHT, DOWN]
                action, game_action = self.generate_action()

                observation = np.append([action], prev_observation)
                done, _, _, _ = self.game.step(game_action)
                data = [observation, 1 - done]
                training_data.append(data)

                if done:
                    break
                else:
                    prev_observation = self.game.is_neighborhood_blocked()

                sleep(TRAIN_CLOCK)

        print(len(training_data))
        return training_data

    def generate_action(self):
        # generate a random action
        action = random.randint(0, 2) - 1

        return action, self.get_game_action(action)

    def get_game_action(self, action):
        # get current direction vector
        snake_direction = game.get_direction_vector()

        new_direction = snake_direction
        if action == -1:
            # turn vector left
            new_direction = [-snake_direction[1], snake_direction[0]]
        elif action == 1:
            # turn vector right
            new_direction = [snake_direction[1], -snake_direction[0]]

        # return Direction corresponding to direction vector
        for pair in self.vector_direction_map:
            if pair[0] == new_direction:
                return pair[1]

    def test_model(self, model):
        steps_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            prev_observation = self.game.is_neighborhood_blocked()
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                    observation = np.append([action], prev_observation)

                    prediction = model.predict(np.array([observation]))
                    predictions.append(prediction)
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(action - 1)
                done, _, _, _ = self.game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    break
                else:
                    prev_observation = self.game.is_neighborhood_blocked()
                    steps += 1
                sleep(TEST_CLOCK)
            steps_arr.append(steps)
        print('Average steps:', mean(steps_arr))
        print(Counter(steps_arr))

    def getModel(self):
        model = Sequential()

        model.add(Dense(units=1, activation="linear", input_shape=(4,), use_bias=False))
        adam = Adam(lr=self.lr)
        model.compile(loss='mse', optimizer=adam, metrics=['mae'])
        model.summary()

        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data])
        y = np.array([i[1] for i in training_data])
        model.fit(X, y, epochs=10, shuffle=True)
        return model

    def train(self):
        self.waitForGUI()
        training_data = self.generate_training_data()
        nn_model = self.getModel()
        nn_model = self.train_model(training_data, nn_model)
        nn_model.save(self.filename)

    def test(self):
        self.waitForGUI()
        nn_model = self.getModel()
        nn_model = load_model(self.filename)
        self.test_model(nn_model)

    def train_and_test(self):
        self.train()
        self.test()

    def waitForGUI(self):
        for i in reversed(range(3)):
            print("Starting in {}".format(i))
            sleep(1)


if __name__ == "__main__":
    try:
        game = Snake()
        nn = SnakeNN(game=game)

        # nn_thread = Thread(name='nn_thread', daemon=True, target=nn.test)
        nn_thread = Thread(name='nn_thread', daemon=True, target=nn.train_and_test)

        nn_thread.start()
        game.run()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()
