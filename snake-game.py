#!/usr/bin/env python3

# Inspired from https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3

# TODO: predisporre la generazione di dati random, etc..

# NOTE: curses methods generally expect FIRST the y/height, and SECOND the x/width

import curses
import traceback
from curses import KEY_DOWN, KEY_EXIT, KEY_LEFT, KEY_RIGHT, KEY_UP
from enum import Enum
from random import randint


class Direction(Enum):
    LEFT, RIGHT, UP, DOWN = range(4)


KEY_ESC = 27


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Snake():
    def __init__(self):
        self.body = []
        self.head = None

    def append(self, point):
        self.body.append(point)
        if not self.head:
            self.head = point

    def prepend(self, point):
        self.body.insert(0, point)
        self.head = self.body[0]

    def removeLast(self):
        self.body.pop()

    def getTail(self):
        return self.body[1:]


class SnakeGame:
    def __init__(self, windowWidth=40, windowHeight=20, wallsEnabled=False):
        self.score = 0
        self.direction = None
        self.windowSize = {"width": windowWidth, "height": windowHeight}
        self.wallsEnabled = wallsEnabled

    def start(self, interactive=True):
        self.interactive = interactive
        if interactive:
            self.initWindow()
        self.initSnake()
        self.generateFood()
        self.draw()
        self.loop()

    def initWindow(self):
        # Initialization of curses
        curses.initscr()

        window = curses.newwin(self.windowSize["height"], self.windowSize["width"], 0, 0)

        curses.noecho()  # disable automatic echoing of keys to the screen
        curses.cbreak()  # react to keys instantly w/o requiring the Enter key to be pressed
        window.keypad(True)  # let curses automatically parse keys and return them as e.g. KEY_DOWN, ...
        curses.curs_set(0)  # 0, 1, or 2, for invisible, normal, or very visible
        window.nodelay(True)  # make getch non-blocking

        # Increases the speed of Snake as its length increases
        # self.window.timeout(round(100 - (len(self.snake.body) / 5 + len(self.snake.body) / 10) % 120))
        window.timeout(120)
        self.window = window

    def generateFood(self):
        food = None
        while food is None:
            # generate food's coordinates
            food = (randint(1, self.windowSize["width"] - 2),
                    randint(1, self.windowSize["height"] - 2))
            if food in self.snake.body:
                food = None
        self.food = Point(food[0], food[1])

    def initSnake(self, initialSize=3):
        head = Point(randint(initialSize, self.windowSize["width"] - 1 - initialSize),
                     randint(initialSize, self.windowSize["height"] - 1 - initialSize))
        self.snake = Snake()
        vertical = randint(0, 1) == 0
        for i in range(initialSize):
            bodyPoint = Point(head.x + i, head.y) if vertical else Point(head.x, head.y + i)
            self.snake.append(bodyPoint)

    def draw(self):
        self.window.clear()
        self.window.border(0)
        self.window.addstr(0, 2, ' Score: ' + str(self.score) + ' ')

        self.window.addch(self.food.y, self.food.x, '*')

        for i, bodyPoint in enumerate(self.snake.body):
            if i == 0:
                self.window.addch(bodyPoint.y, bodyPoint.x, '@')
            else:
                self.window.addch(bodyPoint.y, bodyPoint.x, 'O')

        # if self.direction is not None:
        #     self.window.addstr(self.window_size["height"] - 1,
        #                        2,
        #                        str(self.direction))

    def mapKeyDirection(self, key):
        if key == KEY_LEFT:
            return Direction.LEFT
        elif key == KEY_RIGHT:
            return Direction.RIGHT
        elif key == KEY_UP:
            return Direction.UP
        elif key == KEY_DOWN:
            return Direction.DOWN
        else:
            return None

    def loop(self):
        # if self.interactive:
        key = self.window.getch()
        self.direction = self.mapKeyDirection(key)
        while key == -1:
            key = self.window.getch()
            self.direction = self.mapKeyDirection(key)

        while key != KEY_ESC:
            self.prevDirection = self.direction
            key = self.window.getch()
            self.direction = self.direction if key == -1 else self.mapKeyDirection(key)

            # If SPACE BAR is pressed, wait for another one (Pause/Resume)
            if key == ord(' '):
                while key != ord(' '):
                    key = self.window.getch()
                self.direction = self.prevDirection
                continue

            if self.directionIsInvalid():
                self.direction = self.prevDirection

            # Calculates the new coordinates of the head of the snake. In order to move the snake we must add a point
            # in the next diretion and, if the snake didn't eat, also remove a point from the tail (managed in [1]).
            self.snake.prepend(Point(self.snake.head.x +
                                     (self.direction == Direction.LEFT and -1) +
                                     (self.direction == Direction.RIGHT and 1),
                                     self.snake.head.y +
                                     (self.direction == Direction.UP and -1) +
                                     (self.direction == Direction.DOWN and 1)))

            if self.isCollision():
                break

            # If snake crosses the boundaries, make it enter from the other side
            if self.snake.head.x == 0:
                self.snake.head.x = self.windowSize["width"] - 2
            if self.snake.head.x == self.windowSize["width"] - 1:
                self.snake.head.x = 1
            if self.snake.head.y == 0:
                self.snake.head.y = self.windowSize["height"] - 2
            if self.snake.head.y == self.windowSize["height"] - 1:
                self.snake.head.y = 1

            # When the snake eats the food
            if self.snake.head == self.food:
                self.score += 1
                self.generateFood()
            else:
                # [1]
                self.snake.removeLast()

            self.draw()

        self.terminate()

    def directionIsInvalid(self):
        invalid = False

        # If an invalid key is pressed
        if self.direction not in list(Direction):
            invalid = True

        if self.prevDirection == Direction.UP or self.prevDirection == Direction.DOWN:
            invalid = self.direction == Direction.UP or self.direction == Direction.DOWN
        elif self.prevDirection == Direction.LEFT or self.prevDirection == Direction.RIGHT:
            invalid = self.direction == Direction.LEFT or self.direction == Direction.RIGHT

        return invalid

    def isCollision(self):
        # Exit if snake crosses the boundaries
        if (self.wallsEnabled and
            (self.snake.head.x == 0 or
             self.snake.head.x == self.windowSize["width"] - 1 or
             self.snake.head.y == 0 or
             self.snake.head.y == self.windowSize["height"] - 1)):
            return True

        # If snake runs over itself
        if self.snake.head in self.snake.getTail():
            return True

        return False

    def terminate(self, exception=False):
        # Correctly terminate
        curses.nocbreak()
        self.window.keypad(False)
        curses.echo()

        # Restore teminal to its original operating mode
        curses.endwin()

        if not exception:
            print("Game over")
            print("Score: " + str(self.score))


if __name__ == "__main__":
    game = SnakeGame()
    try:
        game.start(interactive=True)
    except Exception as e:
        game.terminate(True)
        traceback.print_exc()
