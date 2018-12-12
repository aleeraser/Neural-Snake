#!/usr/bin/env python3

# Inspired from https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3

"""
    Snake proceedes randomly. For every step, it observes if the neighbour cells are either
    free or occupied, and it picks a directon, registering whether the outcome was death or not.
    The observation is taken with respect to the relative direction of of the snake.

    Input: 4x1 ([action, left cell, front cell, right cell])
    Output: 1x1 (binary, death or not)

    The idea is to teach the snake to survive.

    Training data is built as follows:
        [[choosen direction, left cell, front cell, right cell], survived]
    Therefore for example:
        • [[-1,  0,  0,  0], 1]: the snake went left while the surrounding cells were all free, and it survived
        • [[0,  0,  1,  0], 0]: the snake went straight while the surrounding cells were all free except for the front one, and it died
        • [[1,  1,  1,  0], 1]: the snake went right while the only free cell was the right one, and it survived
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

TRAIN_CLOCK = 0.003
TEST_CLOCK = 0.003


class SnakeNN:
    def __init__(self, train_games=200, test_games=100, goal_steps=100, lr=1e-2, game=None):
        self.initial_games = train_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.vector_direction_map = [
            [[-1, 0], Direction.LEFT],
            [[0, 1], Direction.UP],
            [[1, 0], Direction.RIGHT],
            [[0, -1], Direction.DOWN]
        ]

        self.game = game

    def generate_training_data(self):
        training_data = []
        steps_arr = []

        for _ in range(self.initial_games):
            steps = 0
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
                    steps += 1

                sleep(TRAIN_CLOCK)

            steps_arr.append(steps)

        self.save_data(message="Do you want to save the collected training data? (y|N)", dataList=training_data, fileName="snake_nn.train")

        print("Average steps:", mean(steps_arr))
        print(Counter(steps_arr))

        print("Number of training data: {}".format(len(training_data)))
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
        debug_info = []

        for _ in range(self.test_games):
            steps = 0
            prev_observation = self.game.is_neighborhood_blocked()
            for i in range(self.goal_steps):
                turn_debug_info = ""
                predictions = []

                # for every possible action predict the probability of surviving
                for action in range(-1, 2):
                    observation = np.append([action], prev_observation)

                    prediction = model.predict(np.array([observation]))
                    predictions.append(prediction)

                    turn_debug_info += "Action {} in state {}. Predicted survival: {}\n".format(action, observation, prediction)

                # choose the action with the best probability to survive
                action = np.argmax(np.array(predictions))

                # -1 because "action" is actually an index starting from 0
                game_action = self.get_game_action(action - 1)

                turn_debug_info += "Decided to take action: {} ({})\n".format(action - 1, game_action)

                done, _, _, _ = self.game.step(game_action)
                if done:
                    turn_debug_info += "--- Dead!\n\n"
                    debug_info.append(turn_debug_info)

                    break
                else:
                    prev_observation = self.game.is_neighborhood_blocked()
                    steps += 1

                if i == 99:
                    turn_debug_info += "Reached 100 steps! Restarting...\n"
                    game.restart()

                # save debug data
                debug_info.append(turn_debug_info)

                sleep(TEST_CLOCK)

            # at the end of the turn save the number of steps performed
            steps_arr.append(steps)

        self.save_data(message="Do you want to save debug data? (y|N)", dataList=debug_info, fileName="snake_nn.debug")

        print("Average steps:", mean(steps_arr))
        print(Counter(steps_arr))

    def save_data(self, message="", model=None, trainData=None, dataList=None, fileName=""):
        if not fileName:
            print("Error, missing file name.")
        elif not message:
            raise Exception

        save_data = input(message)
        if str(save_data).lower() != 'y' and str(save_data).lower() != 'yes':
            return

        if trainData:
            with open(fileName, 'w') as file:
                for data in trainData:
                    file.write("[{} {} {} {}] --> {}\n".format(data[0][0], data[0][1], data[0][2], data[0][3], data[1]))
        elif dataList:
            with open(fileName, 'w') as file:
                for obj in dataList:
                    file.write(str(obj) + "\n")
        elif model:
            model.save(fileName)

        print("Saved in {}!".format(fileName))

    def getModel(self):
        model = Sequential()

        model.add(Dense(units=25, activation="relu", input_shape=(4,), use_bias=False))
        # model.add(Dense(units=1, activation="tanh", use_bias=False))
        model.add(Dense(units=1, activation="sigmoid", use_bias=False))
        adam = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam, metrics=["mae"])
        model.summary()

        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data])
        y = np.array([i[1] for i in training_data])
        model.fit(X, y, epochs=20, shuffle=True)
        return model

    def train(self):
        self.waitForGUI()
        training_data = self.generate_training_data()
        nn_model = self.getModel()
        nn_model = self.train_model(training_data, nn_model)
        self.save_data(message="Do you want to save the trained model? (y|N)", model=nn_model, fileName="snake_nn.model")

    def test(self):
        self.waitForGUI()
        nn_model = self.getModel()
        nn_model = load_model("snake_nn.model")
        self.test_model(nn_model)

    def train_and_test(self):
        self.train()
        self.test()

    def waitForGUI(self):
        for i in reversed(range(3)):
            print("Starting in {}".format(i))
            sleep(1)
        print()


if __name__ == "__main__":
    try:
        game = Snake()
        nn = SnakeNN(game=game)

        # nn_thread = Thread(name="nn_thread", daemon=True, target=nn.train)
        # nn_thread = Thread(name="nn_thread", daemon=True, target=nn.test)
        nn_thread = Thread(name="nn_thread", daemon=True, target=nn.train_and_test)

        nn_thread.start()
        game.run()
    except SystemExit:
        sys.exit()
