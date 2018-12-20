#!/usr/bin/env python3

"""
    Documentation to be updated
"""

# TODO: should not access Snake() object's fields directly

import os
import random
import sys
from collections import Counter, deque
from threading import Thread
from time import sleep

import numpy as np
from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                          initializers)
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from results.logger import Logger

# GAME_CLOCK = 0.004
GAME_CLOCK = 0.008

# TRAIN_GAMES = 30000
# TEST_GAMES = 50
# GOAL_SCORE = 200

# hyperparameters
GAMMA = 0.95
LEARNING_RATE = 1e-2  # 0.01. NOTE: try also 1e-3 and 1e-6

MEMORY_SIZE = 1000000
# BATCH_SIZE = 20
BATCH_SIZE = 50

EXPLORATION_MAX = 1.0
# EXPLORATION_MIN = 0.01
EXPLORATION_MIN = 0.1
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX - EXPLORATION_MIN) / EXPLORATION_STEPS

ACTION_SPACE = 4
LAYERS = 3


class FileNotSavedException(Exception):
    pass


class SnakeNN:
    # def __init__(self, train_games=TRAIN_GAMES, test_games=TEST_GAMES, goal_score=GOAL_SCORE, learning_rate=LEARNING_RATE, game=None):
    def __init__(self, learning_rate=LEARNING_RATE, game=None):
        if game is None:
            raise Exception("Game not connected.")

        self.game = game

        # self.train_games = train_games
        # self.test_games = test_games
        # self.goal_score = goal_score
        self.learning_rate = learning_rate
        # self.vector_direction_map = [
        #     [[-1, 0], Direction.LEFT],
        #     [[0, 1], Direction.UP],
        #     [[1, 0], Direction.RIGHT],
        #     [[0, -1], Direction.DOWN]
        # ]
        self.game_action_map = list(Direction)

        self.epsilon = EXPLORATION_MAX

        self.action_space = ACTION_SPACE
        self.observation_space = (LAYERS, self.game.rows, self.game.cols)
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = None
        self.model_file = "rl.snake_nn.model"
        self.model_loaded = False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)

        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)

        self.epsilon *= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def main_loop(self):
        self.waitForGUI()

        # since we must use multi-threading in order to control kivy, we also need to instantiate
        # the model into the same thread where it will be used, i.e. we cannot do this in __init__
        self.model = self.build_model()
        if os.path.exists(self.model_file):
            self.model = load_model(self.model_file)
            self.model_loaded = True
            print("Loaded model '{}'".format(self.model_file))

        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        logger = Logger("Snake")

        run = 0
        while True:
            try:
                run += 1

                # print("Observation space: {}, Action space: {}".format(self.observation_space, self.action_space))

                # state is a 3 layer 30x30 np matrix
                state, _, _ = self.game.restart()
                # print("State size: {}".format(state.shape))

                # print(state)
                state = np.reshape(state, (1, ) + self.observation_space)
                # state = np.array([1, state])

                step = 0
                while True:
                    sleep(GAME_CLOCK)

                    step += 1
                    # render() ?

                    action = self.act(state)
                    # print(action)

                    state_next, reward, terminal = self.game.step(self.game_action_map[action])
                    if reward == 1:
                        print("Food!")

                    state_next = np.reshape(state_next, (1, ) + self.observation_space)
                    # state_next = np.array([1, state_next])

                    self.remember(state, action, reward, state_next, terminal)
                    state = state_next

                    if terminal:
                        print("Run: {}, eps: {}, steps: {}".format(run, self.epsilon, step))
                        logger.log(step, run)
                        break

                    self.experience_replay()

                if run % 50 == 0:
                    self.save_data(self.model, self.model_file, overwrite=True)

            except KeyboardInterrupt:
                self.save_data(self.model, self.model_file)
                print("Terminating...")
                break

    def save_data(self, model, fileName, message=None, overwrite=False):
        try:
            if not fileName:
                raise Exception("Error, missing file name.")

            if message:
                save_data_input = str(input(message)).lower()
                while save_data_input not in ["y", "yes", "n", "no", ""]:
                    save_data_input = str(input("Wrong input.\n {}".format(message))).lower()

                if save_data_input in ["n", "no", ""]:
                    raise FileNotSavedException

                if os.path.exists(fileName) and not overwrite:
                    overwrite_message = "File '{}' already exists. Overwrite? [y|N] > ".format(fileName)
                    overwrite = input(overwrite_message)
                    while overwrite not in ["y", "yes", "n", "no", ""]:
                        overwrite = input("Wrong input. {}".format(overwrite_message))

                    if overwrite in ["n", "no", ""]:
                        raise FileNotSavedException

            model.save(fileName)

            print("Saved in '{}'.\n".format(fileName))

        except FileNotSavedException:
            print("No data saved.\n")
            return

    def notify(self, title, text):
        os.system("""
            osascript -e 'display notification "{}" with title "{}" sound name "{}"'
            """.format(text, title, "Glass.aiff"))

    def build_model(self):
        model = Sequential()

        # # model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(self.game.cols, self.game.rows, 3)))
        # model.add(Conv2D(32, kernel_size=(8, 8), activation="relu", input_shape=(self.game.cols, self.game.rows, 3)))
        # model.add(Conv2D(64, (3, 3), activation="relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(128, activation="relu"))
        # # model.add(Dense(256, activation="relu"))
        # # model.add(Dense(512, activation="relu"))
        # # model.add(Dropout(0.5))
        # model.add(Dense(3))  # one for each possible action

        # Inspired by DeepMind's Atari NN
        initializer = initializers.random_normal(stddev=0.02)

        # model.add(Conv2D(32, (8, 8), activation="relu", data_format="channels_first",
        #                  strides=(4, 4), kernel_initializer=initializer, padding='same',
        #                  input_shape=(self.layers, self.rows, self.columns)))
        model.add(Conv2D(32,
                         (8, 8),
                         activation="relu",
                         data_format="channels_first",
                         strides=(4, 4),
                         kernel_initializer=initializer,
                         padding='same',
                         input_shape=self.observation_space))
        model.add(Conv2D(64,
                         (4, 4),
                         activation="relu",
                         data_format="channels_first",
                         strides=(2, 2),
                         kernel_initializer=initializer,
                         padding='same'))
        model.add(Conv2D(64,
                         (3, 3),
                         activation="relu",
                         data_format="channels_first",
                         strides=(1, 1),
                         kernel_initializer=initializer,
                         padding='same'))
        model.add(Flatten())
        model.add(Dense(512,
                        activation="relu",
                        kernel_initializer=initializer))
        model.add(Dense(ACTION_SPACE,
                        kernel_initializer=initializer))

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()

        # model._make_predict_function()

        return model

    def waitForGUI(self):
        for i in reversed(range(3)):
            print("Starting in {}".format(i))
            sleep(1)
        print()


if __name__ == "__main__":

    try:
        # import of kivy environment is deferred since kivy automatically loads its components on import
        from snake_game_kivy import Snake, Direction

        game = Snake()
        nn = SnakeNN(game=game)

        nn_thread = Thread(name="nn_thread", daemon=True, target=nn.main_loop)

        nn_thread.start()
        game.run()
    except SystemExit:
        sys.exit()
