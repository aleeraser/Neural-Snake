#!/usr/bin/env python3

# Inspired from https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3

# NOTE: curses methods expect FIRST the y/height, and SECOND the x/width

import argparse
import curses
import logging
import random
import sys
import time
import traceback
from curses import KEY_DOWN, KEY_EXIT, KEY_LEFT, KEY_RIGHT, KEY_UP
from enum import Enum

import numpy as np


class Direction(Enum):
    LEFT, RIGHT, UP, DOWN = range(4)


KEY_ESC = 27
COLS = 30
ROWS = 30
INITIAL_LENGHT = 4

WINDOW_TIMEOUT = 1
GAME_CLOCK = 0.01

direction_vector_map = {
    Direction.LEFT: [-1, 0],
    Direction.UP: [0, -1],
    Direction.RIGHT: [1, 0],
    Direction.DOWN: [0, 1]
}

direction_group = {
    Direction.LEFT: 'horizontal',
    Direction.UP: 'vertical',
    Direction.RIGHT: 'horizontal',
    Direction.DOWN: 'vertical'
}


class SnakeGame:
    def __init__(self):
        # window
        self.cols = COLS
        self.rows = ROWS

        self.logger = self.init_logger()

        # to render or not
        self.render = False
        self.window = None
        self.game_frame = None

        self.direction = random.choice(list(Direction))
        # self.direction = None

        # entities
        self.snake = None
        self.head = None
        self.init_snake()
        self.food = None

        self.lenght = INITIAL_LENGHT

        # stats
        self.score = 0
        self.score_arr = []
        self.steps = 0
        self.steps_arr = []
        self.deaths = 0
        self.start_time = None

        self.done = False
        self.reward = 0

    def init_logger(self):
        logger = logging.getLogger(__file__)
        hdlr = logging.FileHandler(__file__ + ".log")
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.DEBUG)

        logger.info("---------------------------------- init")

        return logger

    def init_game_frame(self):
        # Initialization of curses
        window = curses.initscr()
        game_frame = curses.newwin(self.rows, self.cols, 1, 0)

        curses.start_color()

        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_WHITE)

        curses.noecho()  # disable automatic echoing of keys to the screen
        curses.cbreak()  # react to keys instantly w/o requiring the Enter key to be pressed
        window.keypad(True)  # let curses automatically parse keys and return them as e.g. KEY_DOWN, ...
        curses.curs_set(0)  # 0, 1, or 2, for invisible, normal, or very visible
        window.nodelay(True)  # make getch non-blocking

        # Increases the speed of Snake as its length increases
        # self.window.timeout(round(100 - (len(self.snake.body) / 5 + len(self.snake.body) / 10) % 120))
        window.timeout(WINDOW_TIMEOUT)
        self.window = window
        self.game_frame = game_frame

    def draw(self):
        self.game_frame.clear()
        self.game_frame.border(0)
        self.window.addstr(0, 0, ' Score: {} | Deaths: {} | Steps: {} '.format(self.score, self.deaths, self.steps), curses.color_pair(3))

        self.game_frame.addstr(self.food[1], self.food[0], '•', curses.color_pair(2))

        for body_point in self.snake:
            if body_point == self.head:
                self.game_frame.addstr(body_point[1], body_point[0], '@', curses.color_pair(1))
            else:
                self.game_frame.addstr(body_point[1], body_point[0], 'O', curses.color_pair(1))

        self.game_frame.refresh()

    @property
    def new_food_location(self):
        while True:
            # generate new coords for food until they are valid (i.e. they don't overlap with a snake block)
            food = [random.randint(1, dim - 2) for dim in [self.cols, self.rows]]
            if food not in self.snake and food != self.food:
                return food

    def init_snake(self):
        head = [random.randint(2, dim - 1 - INITIAL_LENGHT) for dim in [self.cols, self.rows]]
        snake = []

        vertical = random.randint(0, 1) == 0
        for i in range(INITIAL_LENGHT):
            body_point = [head[0], head[1] + i] if vertical else [head[0] + i, head[1]]
            snake.append(body_point)

        self.direction = Direction.UP if vertical else Direction.LEFT

        self.head = head
        self.snake = snake

    @property
    def direction_vector(self):
        return direction_vector_map[self.direction]

    def start(self, gui=False):
        self.logger.info("start")
        self.food = self.new_food_location

        key = None

        if gui:
            self.render = True
            self.init_game_frame()
            self.draw()

            key = self.window.getch()

        self.start_time = time.time()

        try:
            while True:
                if self.render:
                    key = self.window.getch()

                    if key == ord('j'):
                        self.step()
                        # self.logger.info("Food: {}, snake: {}".format(self.food, self.snake))
                    elif key == KEY_ESC:
                        break
                    else:
                        self.step()
                else:
                    self.step()
                    # self.logger.info("Food: {}, snake: {}".format(self.food, self.snake))
                    time.sleep(GAME_CLOCK)

                if self.render:
                    self.draw()
        except KeyboardInterrupt:
            pass

        self.terminate()

    def set_direction(self, new_direction):
        if direction_group[new_direction] != direction_group[self.direction]:
            self.direction = new_direction

    def step(self, action=None):
        self.set_direction(action if action else random.choice(list(Direction)))
        return self.move()

    def move(self):
        self.done = False
        self.reward = 0

        # calculate new head coords
        new_head = [sum(x) for x in zip(self.head, direction_vector_map[self.direction])]

        # check for collisions
        if not self.check_in_bounds(new_head) or new_head in self.snake:
            self.done = True
            self.reward = -1

            obs = self.generate_observation()

            self.logger.info("Dead. Score: {}, steps: {}".format(self.score, self.steps))
            self.deaths += 1
            self.reset()

            return obs

        self.steps += 1

        # check for food eaten
        if new_head == self.food:
            self.lenght += 1
            self.score += 1
            self.reward = 1
            self.food = self.new_food_location
            self.logger.info("Food. Score: {}".format(self.score))
        else:
            self.snake.pop()

        self.head = new_head
        self.snake.insert(0, self.head)  # = self.snake[-self.lenght:] + [self.head]

        return self.generate_observation()

    def generate_observation(self):
        # print("Obs: ", self.done, self.score, self.snake, self.food)
        # state, reward, terminal
        return self.get_state(), float(self.reward), self.done

    def get_state(self):
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        # print("Head: {}, Food: {}, Snake: {}.".format(self.head, self.food, self.snake))

        head_m = np.zeros(shape=(self.cols, self.rows))
        food_m = np.zeros(shape=(self.cols, self.rows))
        snake_m = np.zeros(shape=(self.cols, self.rows))

        head_m[self.cols - 1 - self.head[1], self.head[0]] = 1
        food_m[self.cols - 1 - self.food[1], self.food[0]] = 1
        for coord in self.snake:
            if coord != self.head:
                snake_m[self.cols - 1 - coord[1], coord[0]] = 1

        # print("• Head:\n{}\n\n\n• Food:\n{}\n\n\n• Snake:\n{}\n.".format(head_m, food_m, snake_m))

        return np.array([head_m, food_m, snake_m])

    def check_in_bounds(self, pos):
        return all(1 <= pos[x] < dim - 1 for x, dim in enumerate([COLS, ROWS]))

    def reset(self):
        # if 'reset' wasn't called because of a death
        if self.reward != -1:
            self.logger.info("Reset. Score: {}, steps: {}\n".format(self.score, self.steps))

        self.snake.clear()
        self.lenght = INITIAL_LENGHT

        # reset score
        self.score_arr.append(self.score)
        self.score = 0
        self.steps_arr.append(self.steps)
        self.steps = 0

        # generates new entities
        self.init_snake()
        self.food = self.new_food_location

        return self.generate_observation()

    def terminate(self):
        if self.render:
            # Correctly terminate
            curses.nocbreak()
            self.game_frame.keypad(False)
            curses.echo()

            # Restore teminal to its original operating mode
            curses.endwin()

        self.logger.info("Stats:")
        self.logger.info("• avg score: {}".format(round(sum(self.score_arr) / self.deaths)))
        self.logger.info("• max score: {}".format(max(self.score_arr)))
        self.logger.info("• deaths: {}".format(self.deaths))
        self.logger.info("• avg steps: {}".format(round(sum(self.steps_arr) / self.deaths)))
        self.logger.info("• max steps: {}".format(max(self.steps_arr)))
        self.logger.info("• execution time: {} s".format(round(time.time() - self.start_time, 2)))
        self.logger.info("---------------------------------- end")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', type=bool, default=False)
    args = parser.parse_args()

    try:
        game = SnakeGame()

        game.start(gui=args.gui)
    except Exception as e:
        game.terminate()
        traceback.print_exc()
