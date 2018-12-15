#!/usr/bin/env python3

# Inspired from https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3

"""
    Snake proceedes randomly. For every step, it observes if the neighbour cells are either
    free or occupied, and also observers the angle between its current direction and the food.
    It then picks a directon, whose outcome is defined in this way:
    • -1 if the snake is dead
    • 0 if the snake is not dead, but it picked the wrong direction (i.e. w.r.t. the food)
    • 1 if the snake is not dead and the right direction was picked

    The observation is taken with respect to the relative direction of of the snake.

    Input: 5x1 ([action, left cell, front cell, right cell, angle between current direction and food])
    Output: 1x1, probability of the snake to survive after performing the given action.

    The idea is to teach the snake to survive and to minimize its distance to the food.

    Training data is built as follows:
        [[choosen direction, left cell, front cell, right cell, angle], outcome]
"""

# TODO: should not access Snake() object's fields directly

import math
import os
import random
import sys
from collections import Counter
from statistics import mean
from threading import Thread
from time import sleep

import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam

TRAIN_CLOCK = 0.004
TEST_CLOCK = 0.03

# TRAIN_GAMES = 30000
TRAIN_GAMES = 1
# TEST_GAMES = 50
TEST_GAMES = 1
GOAL_SCORE = 200
LEARNING_RATE = 1E-2


class FileNotSavedException(Exception):
    pass


class SnakeNN:
    def __init__(self, game=None):
        self.train_games = TRAIN_GAMES
        self.test_games = TEST_GAMES
        self.goal_score = GOAL_SCORE
        self.learning_rate = LEARNING_RATE
        self.vector_direction_map = [
            [[-1, 0], Direction.LEFT],
            [[0, 1], Direction.UP],
            [[1, 0], Direction.RIGHT],
            [[0, -1], Direction.DOWN]
        ]

        if game is None:
            raise Exception("Game not connected.")

        self.game = game

    def generate_training_data(self):
        training_data = []
        steps_log = []
        scores_log = []

        for _ in range(self.train_games):
            steps = 0

            # observation from snake game
            _, prev_score, _, _ = self.game.generate_observation()

            # observation from nn agent
            prev_observation = self.generate_observation()

            prev_food_distance = self.get_food_distance()

            while True:
                steps += 1

                # action: -1 (turn left), 0 (straight), 1 (turn right)
                # game action: Direction.[LEFT, UP, RIGHT, DOWN]
                action, game_action = self.generate_action()

                observation = np.append([action], prev_observation)
                done, score, _, _ = self.game.step(game_action)

                if done:
                    # -1: snake is dead
                    training_data.append([observation, -1])
                    break
                else:
                    if score == self.goal_score:
                        training_data.append([observation, 1])
                        break

                    food_distance = self.get_food_distance()
                    if score > prev_score or food_distance < prev_food_distance:
                        # 1: score increased or distance to food decreased
                        training_data.append([observation, 1])
                    else:
                        # 0: snake is alive but chose wrong direction
                        training_data.append([observation, 0])

                    prev_observation = self.generate_observation()
                    prev_food_distance = food_distance

                sleep(TRAIN_CLOCK)

            steps_log.append(steps)
            scores_log.append(score)

        self.save_data(message="Save the collected training data? (y|N) > ", dataList=training_data, fileName="snake_nn.train")

        print("Average steps: {}, max: {}".format(mean(steps_log), max(steps_log)))
        print(Counter(steps_log))
        print("Average score: {}, max: {}".format(mean(scores_log), max(scores_log)))
        print(Counter(scores_log))

        print("Number of training data: {}".format(len(training_data)))
        return training_data

    def get_food_direction_vector(self):
        return np.array(self.game.food) - np.array(self.game.head)

    def get_food_distance(self):
        return np.linalg.norm(self.get_food_direction_vector())

    def generate_action(self):
        # generate a random action
        action = random.randint(0, 2) - 1

        return action, self.get_game_action(action)

    def get_game_action(self, action):
        # get current direction vector
        snake_direction = self.game.direction_vector

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

    def generate_observation(self):
        food_direction = self.get_food_direction_vector()
        food_angle = self.get_angle(self.game.direction_vector, food_direction)

        return np.append(self.is_neighborhood_blocked(), food_angle)

    def is_neighborhood_blocked(self):
        direction_vector = self.game.direction_vector

        # turn vector left
        left = [sum(x) for x in zip(self.game.head, [-direction_vector[1], direction_vector[0]])]
        # keep direction
        front = [sum(x) for x in zip(self.game.head, direction_vector)]
        # turn vector right
        right = [sum(x) for x in zip(self.game.head, [direction_vector[1], -direction_vector[0]])]

        directions = [left, front, right]

        return [int(not self.game.check_in_bounds(direction) or direction in self.game.snake) for direction in directions]

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def test_model(self, model):
        steps_log = []
        scores_log = []
        debug_info = []
        retraining_data = []

        for _ in range(self.test_games):
            steps = 0
            score = 0

            # observation from snake game
            _, prev_score, snake, food = self.game.generate_observation()

            # observation from nn agent
            prev_observation = self.generate_observation()

            prev_food_distance = self.get_food_distance()

            i = 0
            while True:
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

                observation = np.append([action], prev_observation)
                done, score, snake, food = self.game.step(game_action)

                i += 1

                if done:
                    turn_debug_info += "--- Dead!\n\n"
                    turn_debug_info += "\tsteps: {}\n".format(steps)
                    turn_debug_info += "\tsnake: {}\n".format(snake)
                    turn_debug_info += "\tfood: {}\n".format(food)
                    turn_debug_info += "\tprev_observation: {}\n".format(prev_observation)
                    turn_debug_info += "\tpredictions: {}\n".format(predictions)
                    debug_info.append(turn_debug_info)

                    # -1: snake is dead
                    retraining_data.append([observation, -1])

                    break
                else:
                    if score == self.goal_score:
                        retraining_data.append([observation, 1])
                        break

                    food_distance = self.get_food_distance()
                    if score > prev_score or food_distance < prev_food_distance:
                        # 1: score increased or distance to food decreased
                        retraining_data.append([observation, 1])
                    else:
                        # 0: snake is alive but chose wrong direction
                        retraining_data.append([observation, 0])

                    prev_observation = self.generate_observation()
                    prev_food_distance = food_distance

                    steps += 1

                # save debug data
                debug_info.append(turn_debug_info)

                sleep(TEST_CLOCK)

            # at the end of the turn save the number of steps and score achieved
            steps_log.append(steps)
            scores_log.append(score)

        self.save_data(message="Save debug data? (y|N) > ", dataList=debug_info, fileName="snake_nn.debug")

        print("Average steps: {}, max: {}".format(mean(steps_log), max(steps_log)))
        print(Counter(steps_log))
        print("Average score: {}, max: {}".format(mean(scores_log), max(scores_log)))
        print(Counter(scores_log))

        print("Number of (re)training data: {}".format(len(retraining_data)))
        return retraining_data

    def save_data(self, message="", model=None, debugTrainData=None, dataList=None, fileName=""):
        try:
            if not fileName:
                raise Exception("Error, missing file name.")
            elif not message:
                raise Exception("No message specified.")

            save_data_input = str(input(message)).lower()
            while save_data_input not in ["y", "yes", "n", "no", ""]:
                save_data_input = str(input("Wrong input.\n {}".format(message))).lower()

            if save_data_input in ["n", "no", ""]:
                raise FileNotSavedException

            if os.path.exists(fileName):
                overwrite_message = "File '{}' already exists. Overwrite? [y|N] > ".format(fileName)
                overwrite = input(overwrite_message)
                while overwrite not in ["y", "yes", "n", "no", ""]:
                    overwrite = input("Wrong input. {}".format(overwrite_message))

                if overwrite in ["n", "no", ""]:
                    raise FileNotSavedException

            if debugTrainData and not dataList and not model:
                with open(fileName, 'w') as file:
                    for data in debugTrainData:
                        file.write("[{} {} {} {}] --> {}\n".format(data[0][0], data[0][1], data[0][2], data[0][3], data[1]))
            elif dataList and not debugTrainData and not model:
                with open(fileName, 'w') as file:
                    for obj in dataList:
                        file.write(str(obj) + "\n")
            elif model and not dataList and not debugTrainData:
                model.save(fileName)
            else:
                raise Exception("Too many arguments given. '" + fileName + "' not saved.")

            print("Saved in '{}'.\n".format(fileName))

        except FileNotSavedException:
            print("No data saved.\n")
            return

    def getModel(self):
        model = Sequential()

        model.add(Dense(units=25, activation="relu", input_shape=(5,), use_bias=False))
        # model.add(Dense(units=1, activation="tanh", use_bias=False))
        model.add(Dense(units=1, activation="sigmoid", use_bias=False))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam, metrics=["mae"])
        model.summary()

        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data])
        y = np.array([i[1] for i in training_data])
        model.fit(X, y, epochs=20, shuffle=True)
        return model

    def train(self, training_data=None):
        self.waitForGUI()
        if not training_data:
            training_data = self.generate_training_data()
        nn_model = self.getModel()
        nn_model = self.train_model(training_data, nn_model)

        print("Training finished.")

        self.save_data(message="Save the trained model? (y|N) > ", model=nn_model, fileName="snake_nn.model")

    def test(self):
        self.waitForGUI()
        nn_model = self.getModel()
        nn_model = load_model("snake_nn.model")
        training_data = self.test_model(nn_model)

        retrain_message = "Use collected data to re-train the neural network? [y|N] > "
        retrain_input = str(input(retrain_message)).lower()
        while retrain_input not in ["y", "yes", "n", "no", ""]:
            retrain_input = str(input("Wrong input.\n {}").format(retrain_message)).lower()

        if retrain_input in ["y", "yes"]:
            self.train(training_data)

        print("Testing finished.")

    def train_and_test(self):
        self.train()
        self.test()

    def waitForGUI(self):
        for i in reversed(range(3)):
            print("Starting in {}".format(i))
            sleep(1)
        print()


if __name__ == "__main__":
    mode_selection_message = "Type:\n• '1' for training\n• '2' for testing\n• '3' for both\n> "
    mode = input(mode_selection_message)
    while mode not in ["1", "2", "3"]:
        mode = input("Wrong input. {}".format(mode_selection_message))

    try:
        # import of kivy environment is deferred since kivy automatically loads its components on import
        from snake_game_kivy import Snake, Direction

        game = Snake()
        nn = SnakeNN(game=game)

        if mode == "1":
            nn_thread = Thread(name="nn_thread", daemon=True, target=nn.train)
        elif mode == "2":
            nn_thread = Thread(name="nn_thread", daemon=True, target=nn.test)
        elif mode == "3":
            nn_thread = Thread(name="nn_thread", daemon=True, target=nn.train_and_test)

        nn_thread.start()
        game.run()
    except SystemExit:
        sys.exit()
