#!/usr/bin/env python3

# Inspired from https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3

# TODO: hai reso il gioco OO e strutturato. Devi finire di fare il refactoring di loop() e spezzarla in metodi più semplici.
# Dopodichè devi predisporre la generazione di dati random, etc..

import curses
from curses import KEY_DOWN, KEY_EXIT, KEY_LEFT, KEY_RIGHT, KEY_UP
from random import randint

KEY_ESC = 27


class SnakeGame:
    def __init__(self, window_width=20, window_height=40, walls_enabled=False):
        self.score = 0
        self.key = None
        self.window_size = {"width": window_width, "height": window_height}
        self.walls_enabled = walls_enabled

    def start(self):
        self.init_window()
        self.init_snake()  # snake = [[4, 10], [4, 9], [4, 8]]
        self.generate_food()  # food = [10, 20]
        self.draw()
        self.loop()

    def init_window(self):
        # Initialization of curses
        curses.initscr()

        window = curses.newwin(self.window_size["width"], self.window_size["height"], 0, 0)

        curses.noecho()  # disable automatic echoing of keys to the screen
        curses.cbreak()  # react to keys instantly w/o requiring the Enter key to be pressed
        window.keypad(True)  # let curses automatically parse keys and return them as e.g. KEY_DOWN, ...
        curses.curs_set(0)  # 0, 1, or 2, for invisible, normal, or very visible
        window.nodelay(True)  # make getch non-blocking
        # window.border(0)

        # Increases the speed of Snake as its length increases
        # self.window.timeout(round(100 - (len(self.snake) / 5 + len(self.snake) / 10) % 120))
        window.timeout(120)
        self.window = window

    def generate_food(self):
        food = None
        while food is None:
            # generate food's coordinates
            food = [randint(1, self.window_size["width"] - 2), randint(1, self.window_size["height"] - 2)]
            if food in self.snake:
                food = None
        self.food = food

    def init_snake(self, initial_size=3):
        head_x = randint(initial_size, self.window_size["width"] - initial_size)
        head_y = randint(initial_size, self.window_size["height"] - initial_size)
        self.snake = []
        vertical = randint(0, 1) == 0
        for i in range(initial_size):
            body_point = [head_x + i, head_y] if vertical else [head_x, head_y + i]
            self.snake.insert(0, body_point)

    def draw(self):
        self.window.clear()
        self.window.border(0)
        self.window.addstr(0, 2, ' Score: ' + str(self.score) + ' ')

        self.window.addch(self.food[0], self.food[1], '*')

        for i, body_point in enumerate(self.snake):
            if i == 0:
                self.window.addch(body_point[0], body_point[1], '@')
            else:
                self.window.addch(body_point[0], body_point[1], 'O')

    def loop(self):
        while self.key is None:
            self.key = self.window.getch()

            # If an invalid key is pressed
            if self.key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_ESC]:
                self.key = None

        try:
            while self.key != KEY_ESC:
                self.prevKey = self.key
                newKey = self.window.getch()
                self.key = self.key if newKey == -1 else newKey

                # If SPACE BAR is pressed, wait for another one (Pause/Resume)
                if self.key == ord(' '):
                    self.key = -1
                    while self.key != ord(' '):
                        self.key = self.window.getch()
                    self.key = self.prevKey
                    continue

                # If an invalid key is pressed
                if self.key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_ESC] or self.directionIsInvalid():
                    self.key = self.prevKey

                # Calculates the new coordinates of the head of the snake. In order to move the snake we must add a point
                # in the next diretion and, if the snake didn't eat, also remove a point from the tail. Done in (1).
                self.snake.insert(0, [self.snake[0][0] + (self.key == KEY_DOWN and 1) + (self.key == KEY_UP and -1), self.snake[0][1] + (self.key == KEY_LEFT and -1) + (self.key == KEY_RIGHT and 1)])

                # Exit if snake crosses the boundaries
                if self.walls_enabled and (self.snake[0][0] == 0 or self.snake[0][0] == self.window_size["width"] - 1 or self.snake[0][1] == 0 or self.snake[0][1] == self.window_size["height"] - 1):
                    break

                # If snake crosses the boundaries, make it enter from the other side
                if self.snake[0][0] == 0:
                    self.snake[0][0] = self.window_size["width"] - 2
                if self.snake[0][1] == 0:
                    self.snake[0][1] = self.window_size["height"] - 2
                if self.snake[0][0] == self.window_size["width"] - 1:
                    self.snake[0][0] = 1
                if self.snake[0][1] == self.window_size["height"] - 1:
                    self.snake[0][1] = 1

                # If snake runs over itself
                if self.snake[0] in self.snake[1:]:
                    break

                # When snake eats the food
                if self.snake[0] == self.food:
                    self.score += 1
                    self.generate_food()
                else:
                    # (1)
                    self.snake.pop()

                self.draw()

        except Exception as e:
            self.terminate(e)

        self.terminate()

    def directionIsInvalid(self):
        if self.prevKey == KEY_UP or self.prevKey == KEY_DOWN:
            return self.key == KEY_UP or self.key == KEY_DOWN
        elif self.prevKey == KEY_LEFT or self.prevKey == KEY_RIGHT:
            return self.key == KEY_LEFT or self.key == KEY_RIGHT

    def terminate(self, exception=None):
        # Correctly terminate
        curses.nocbreak()
        self.window.keypad(False)
        curses.echo()

        # Restore teminal to its original operating mode
        curses.endwin()

        if exception is not None:
            print(exception)

        print("Game over")
        print("Score: " + str(self.score))


if __name__ == "__main__":
    game = SnakeGame()
    game.start()
