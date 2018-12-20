#!/usr/bin/env python3

"""
    Documentation to be updated
"""

# TODO: should not access Snake() object's fields directly

import argparse
import csv
import os
import pickle
import random
import sys
import traceback
from collections import Counter, deque
from threading import Thread
from time import sleep

import numpy as np
from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                          initializers)
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from snake_game_terminal import Direction, SnakeGame

# GAMMA = 0.99
GAMMA = 0.95
LEARNING_RATE = 1e-6  # NOTE: try also 1e-3 and 1e-2 for faster learning

MEMORY_SIZE = 100000
MIN_MEMORY_LENGTH = 5000
BATCH_SIZE = 32

EPS_MAX = 1.0
# EPS_MIN = 0.1
EPS_MIN = 0.01
EPS_TEST = 0.02
# EPS_STEPS = 500000
EPS_STEPS = 85000
EPS_DECAY = (EPS_MAX - EPS_MIN) / EPS_STEPS

ACTION_SPACE = 4
LAYERS = 3

BYTES_MAX = 2**31 - 1


class SnakeNN:
    def __init__(self, learning_rate=LEARNING_RATE):
        self.game = SnakeGame()

        self.learning_rate = learning_rate
        self.game_action_map = list(Direction)

        self.epsilon = EPS_MAX

        self.action_space = ACTION_SPACE
        self.observation_space = (LAYERS, self.game.rows, self.game.cols)

        # FIFO
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = self.build_model()
        self.load_progress()

    def predict(self, state):
        # eps-greedy
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)

        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def act(self, action_index):
        action = self.game_action_map[action_index]
        new_state, reward, done = self.game.step(action)

        # self.game.logger.info("Action: {},\treward: {}".format(action, reward))

        return new_state, reward, done

    def observe(self, state):
        batch = random.sample(self.memory, BATCH_SIZE)
        input_states = np.zeros((BATCH_SIZE,) + state.shape[1:])
        q_values = np.zeros((BATCH_SIZE, ACTION_SPACE))

        # experience replay
        for i in range(BATCH_SIZE):
            state = batch[i][0]
            action_index = batch[i][1]
            reward = batch[i][2]
            new_state = batch[i][3]
            done = batch[i][4]

            input_states[i] = state
            q_values[i] = self.model.predict(state)

            if done:
                q_values[i, action_index] = reward
            else:
                q_update = self.model.predict(new_state)  # [0]
                q_values[i, action_index] = reward + GAMMA * np.max(q_update)

        loss = self.model.train_on_batch(input_states, q_values)
        # self.game.logger.info("Loss for this iteration: {}".format(loss))
        return loss

    def remember(self, state, action, reward, next_state, done):
        self.memory.appendleft((state, action, reward, next_state, done))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop()

    def train(self, gui):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        if gui:
            self.game.start(gui)

        run = 0
        while True:
            if os.path.exists("terminate"):
                break

            try:
                run += 1

                # state is a 3 layer 30x30 np matrix
                state, _, _ = self.game.reset()

                state = np.reshape(state, (1, ) + self.observation_space)

                step = 0
                while True:
                    step += 1

                    action_index = self.predict(state)
                    new_state, reward, done = self.act(action_index)

                    # self.game.logger.info("State: {}".format(new_state))
                    # self.game.logger.info("Action: {}, reward: {}".format(self.game_action_map[action_index], reward))

                    if gui:
                        self.game.draw()

                    new_state = np.reshape(new_state, (1, ) + self.observation_space)
                    self.remember(state, action_index, reward, new_state, done)

                    if len(self.memory) >= MIN_MEMORY_LENGTH:
                        self.observe(new_state)

                        # eps annealing
                        self.epsilon -= EPS_DECAY
                        self.epsilon = max(EPS_MIN, self.epsilon)

                    state = new_state

                    if done:
                        self.game.logger.info("Run: {}, eps: {}, steps: {}".format(run, self.epsilon, step))
                        break

                if run % 5000 == 0:
                    self.save_progress(run)

            except Exception:
                self.game.terminate()
                self.game = None
                traceback.print_exc()
                break

        if self.game:
            self.game.terminate()
            self.game = None

        self.save_progress(run)
        print("Terminating...")

    def test(self, gui):
        # no random actions
        self.epsilon = 0

        if gui:
            self.game.start(gui)

        # state is a 3 layer 30x30 np matrix
        state, _, _ = self.game.reset()
        state = np.reshape(state, (1, ) + self.observation_space)

        try:
            while not os.path.exists("terminate"):
                action_index = self.predict(state)
                state, _, _ = self.act(action_index)
                state = np.reshape(state, (1, ) + self.observation_space)
                if gui:
                    self.game.draw()
                    sleep(0.07)
        except KeyboardInterrupt:
            pass

        if self.game:
            self.game.terminate()
            self.game = None

    def save_progress(self):
        if not os.path.exists("model"):
            os.makedirs("model")

        self.model.save_weights("model/weights.h5", overwrite=True)
        print("Saved weights")

        # workaround for pickle file size limitation on OSX
        bytes_out = pickle.dumps(self.memory)
        with open("model/memory", "wb") as memory_file:
            for idx in range(0, len(bytes_out), BYTES_MAX):
                memory_file.write(bytes_out[idx:idx + BYTES_MAX])
            print("Saved memory")

        with open("model/parameters.csv", "w") as parameters:
            writer = csv.writer(parameters)
            writer.writerow(["epsilon", self.epsilon])

        print("Saved model")

    def load_progress(self):
        if os.path.isfile("model/weights.h5"):
            self.model.load_weights("model/weights.h5")
            print("Loaded weights")

        # workaround for pickle file size limitation on OSX
        if os.path.isfile("model/memory"):
            bytes_in = bytearray(0)
            input_size = os.path.getsize("model/memory")
            with open("model/memory", 'rb') as memory_file:
                for _ in range(0, input_size, BYTES_MAX):
                    bytes_in += memory_file.read(BYTES_MAX)
            self.memory = pickle.loads(bytes_in)
            print("Loaded memory")

        if os.path.isfile("model/parameters.csv"):
            with open("model/parameters.csv") as parameters:
                reader = csv.reader(parameters)
                for row in reader:
                    try:
                        self.epsilon = float(row[1])
                    except ValueError:
                        print("Cannot load parameter '{}' with value '{}'".format(row[0], row[1]))

        print("Loaded model")

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


if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-g", "--gui", action="store_true", default=False)
        parser.add_argument("-m", "--mode", choices=["train", "test"], default="train")
        args = parser.parse_args()

        nn = SnakeNN()
        if args.mode == "train":
            nn.train(gui=args.gui)
        else:
            nn.test(gui=args.gui)

    except Exception:
        sys.exit()
