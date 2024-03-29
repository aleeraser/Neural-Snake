#!/usr/bin/env python3

# Inspired from https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3

# NOTE: curses methods expect FIRST the y/height, and SECOND the x/width

import curses
import random
import sys
import traceback
from curses import KEY_DOWN, KEY_EXIT, KEY_LEFT, KEY_RIGHT, KEY_UP
from enum import Enum


class CollisionException(Exception):
    pass


class InvalidDirectionException(Exception):
    pass


class Direction(Enum):
    LEFT, RIGHT, UP, DOWN = range(4)


KEY_ESC = 27


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def equals(self, point):
        return self.x == point.x and self.y == point.y


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

    def toString(self):
        return str([str(point) for point in self.body])

    def __contains__(self, point):
        for bodyPoint in self.body:
            if bodyPoint.equals(point):
                return True
        return False


class SnakeGame:
    def __init__(self, windowWidth=50, windowHeight=20, wallsEnabled=True):
        self.windowSize = {"width": windowWidth, "height": windowHeight}
        self.wallsEnabled = wallsEnabled
        self.debug = None

    def start(self, interactive=True):
        self.score = 0

        self.initWindow()

        self.interactive = interactive
        self.paused = False

        key = self.window.getch()

        while key != KEY_ESC:
            self.setup()
            self.draw()

            try:
                self.loop()

                # if we are here, loop terminated because of a KEY_ESC
                break
            except CollisionException as e:
                self.window.addstr(self.windowSize["height"] - 1, round(self.windowSize["width"] / 2 - len(' Game Over ') / 2) - 1, ' Game Over ')
                key = -1

                while key == -1:
                    key = self.window.getch()

        self.terminate()

    def setup(self):
        self.score = 0
        self.window.clear()
        self.initSnake(initialSize=8)
        self.generateFood()
        self.draw()

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
            food = Point(random.randint(1, self.windowSize["width"] - 2),
                         random.randint(1, self.windowSize["height"] - 2))
            if food in self.snake:
                food = None
        self.food = food

    def initSnake(self, initialSize=3):
        head = Point(random.randint(initialSize, self.windowSize["width"] - 1 - initialSize),
                     random.randint(initialSize, self.windowSize["height"] - 1 - initialSize))
        self.snake = Snake()
        vertical = random.randint(0, 1) == 0
        for i in range(initialSize):
            bodyPoint = Point(head.x, head.y + i) if vertical else Point(head.x + i, head.y)
            self.snake.append(bodyPoint)

        self.prevDirection = Direction.UP if vertical else Direction.LEFT

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

        if self.debug is not None:
            self.window.addstr(1, 2, str(self.debug))

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
            return self.prevDirection

    def loop(self):
        self.window.addstr(self.windowSize["height"] - 1, 12, ' Press any key to start ')

        key = self.window.getch()
        direction = None

        if self.interactive:
            direction = self.mapKeyDirection(key)
            while key == -1:
                key = self.window.getch()
                direction = self.mapKeyDirection(key)

        if self.directionIsInvalid(direction):
            direction = self.prevDirection

        while key != KEY_ESC:
            self.prevDirection = direction
            key = self.window.getch()

            if key == ord('r'):
                self.interactive = not self.interactive
                key = -1

            if self.paused:
                self.window.addstr(self.windowSize["height"] - 1, 8, ' Paused, press spacebar to resume ')
                # If the game is paused wait for a SPACE BAR to resume
                if key == ord(' '):
                    self.paused = False
                continue
            else:
                # If SPACE BAR is pressed pause the game
                if key == ord(' '):
                    self.paused = True
                    continue

            if self.interactive:
                direction = direction if key == -1 else self.mapKeyDirection(key)
            else:
                direction = random.choice(list(Direction))

            try:
                self.step(direction)
            except CollisionException as e:
                raise e
            except InvalidDirectionException as e:
                direction = self.prevDirection
                self.step(direction)

    def step(self, direction=None):
        if direction is None:
            raise Exception("No action specified.")

        if self.directionIsInvalid(direction):
            raise InvalidDirectionException

        # Calculates the new coordinates of the head of the snake. In order to move the snake we must add a point
        # in the next diretion and, if the snake didn't eat, also remove a point from the tail (managed in [1]).
        nextHead = Point(self.snake.head.x +
                         (direction == Direction.LEFT and -1) +
                         (direction == Direction.RIGHT and 1),
                         self.snake.head.y +
                         (direction == Direction.UP and -1) +
                         (direction == Direction.DOWN and 1))

        if not self.wallsEnabled:
            # If snake crosses the boundaries, make it enter from the other side
            if nextHead.x == 0:
                nextHead.x = self.windowSize["width"] - 2
            if nextHead.x == self.windowSize["width"] - 1:
                nextHead.x = 1
            if nextHead.y == 0:
                nextHead.y = self.windowSize["height"] - 2
            if nextHead.y == self.windowSize["height"] - 1:
                nextHead.y = 1

        # When the snake eats the food
        if nextHead.equals(self.food):
            self.score += 1
            self.generateFood()
        else:
            # [1]
            self.snake.removeLast()

        if self.isCollision(nextHead):
            raise CollisionException

        self.snake.prepend(nextHead)

        self.draw()

    def directionIsInvalid(self, direction):
        # If an invalid key is pressed
        if direction not in list(Direction):
            return True

        if self.prevDirection == Direction.UP:
            return direction == Direction.DOWN
        elif self.prevDirection == Direction.DOWN:
            return direction == Direction.UP
        elif self.prevDirection == Direction.LEFT:
            return direction == Direction.RIGHT
        elif self.prevDirection == Direction.RIGHT:
            return direction == Direction.LEFT

    def isCollision(self, nextHead):
        # Exit if snake crosses the boundaries
        if (self.wallsEnabled and
            (nextHead.x == 0 or
             nextHead.x == self.windowSize["width"] - 1 or
             nextHead.y == 0 or
             nextHead.y == self.windowSize["height"] - 1)):
            return True

        # If snake runs over itself
        if nextHead in self.snake:
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

    interactive = True
    if len(sys.argv) > 1:
        if str(sys.argv[1]) == "-r":
            interactive = False

    try:
        game.start(interactive=interactive)
    except Exception as e:
        game.terminate(True)
        traceback.print_exc()
