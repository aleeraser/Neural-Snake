#!/usr/bin/env python3

# Inspired from https://towardsdatascience.com/today-im-going-to-talk-about-a-small-practical-example-of-using-neural-networks-training-one-to-6b2cbd6efdb3

import curses
from curses import KEY_DOWN, KEY_EXIT, KEY_LEFT, KEY_RIGHT, KEY_UP
from random import randint

# Initialization of curses
curses.initscr()

# Constants
KEY_ESC = 27
WINDOW_LINES = 20
# WINDOW_LINES = curses.LINES
WINDOW_COLS = 40
# WINDOW_COLS = curses.COLS
WALLS_ENABLED = False

# Init values
key = KEY_RIGHT
score = 0
snake = [[4, 10], [4, 9], [4, 8]]  # snake coordinates
food = [10, 20]  # food coordinates


def init():
    window = curses.newwin(WINDOW_LINES, WINDOW_COLS, 0, 0)

    curses.noecho()  # disable automatic echoing of keys to the screen
    curses.cbreak()  # react to keys instantly w/o requiring the Enter key to be pressed
    window.keypad(True)  # let curses automatically parse keys and return them as e.g. KEY_DOWN, ...
    curses.curs_set(0)
    window.border(0)
    window.nodelay(1)

    # Display/prints the food
    window.addch(food[0], food[1], '*')

    return window


def directionIsInvalid(prevKey, key):
    if prevKey == KEY_UP or prevKey == KEY_DOWN:
        return key == KEY_DOWN or key == KEY_UP
    elif prevKey == KEY_LEFT or prevKey == KEY_RIGHT:
        return key == KEY_LEFT or key == KEY_RIGHT


def game_loop(window):
    global key
    global score
    global snake
    global food

    # While Esc key is not pressed
    while key != KEY_ESC:
        window.border(0)

        # Printing 'Score'
        window.addstr(0, 2, ' Score: ' + str(score) + ' ')

        # Increases the speed of Snake as its length increases
        window.timeout(round(100 - (len(snake) / 5 + len(snake) / 10) % 120))

        # Previous key pressed
        prevKey = key
        event = window.getch()
        key = key if event == -1 else event

        # If SPACE BAR is pressed, wait for another one (Pause/Resume)
        if key == ord(' '):
            key = -1
            while key != ord(' '):
                key = window.getch()
            key = prevKey
            continue

        # If an invalid key is pressed
        if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_ESC] or directionIsInvalid(prevKey, key):
            key = prevKey

        # Calculates the new coordinates of the head of the snake. In order to move the snake we must add a point
        # in the next diretion and, if the snake didn't eat, also remove a point from the tail. This is taken care of later at [1].
        snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1), snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])

        # Exit if snake crosses the boundaries
        if WALLS_ENABLED and (snake[0][0] == 0 or snake[0][0] == WINDOW_LINES - 1 or snake[0][1] == 0 or snake[0][1] == WINDOW_COLS - 1):
            break

        # If snake crosses the boundaries, make it enter from the other side
        if snake[0][0] == 0:
            snake[0][0] = WINDOW_LINES - 2
        if snake[0][1] == 0:
            snake[0][1] = WINDOW_COLS - 2
        if snake[0][0] == WINDOW_LINES - 1:
            snake[0][0] = 1
        if snake[0][1] == WINDOW_COLS - 1:
            snake[0][1] = 1

        # If snake runs over itself (excluding pressing the key corresponding to the opposite direction)
        if snake[0] in snake[1:]:
            break

        # When snake eats the food
        if snake[0] == food:
            food = None
            score += 1
            while food is None:
                # Calculating next food's coordinates
                food = [randint(1, WINDOW_LINES - 2), randint(1, WINDOW_COLS - 2)]
                if food in snake:
                    food = None
            window.addch(food[0], food[1], '*')
        else:
            # [1] If it does not eat the food, length decreases
            last = snake.pop()
            window.addch(last[0], last[1], ' ')
        window.addch(snake[0][0], snake[0][1], '@')


def terminate(window, exception=None):
    # Correctly terminate curses app
    curses.nocbreak()
    window.keypad(False)
    curses.echo()

    # Restore teminal to its original operating mode
    curses.endwin()

    if exception is not None:
        print(exception)

    print("\nScore - " + str(score))


if __name__ == "__main__":
    try:
        window = init()
        game_loop(window)
        terminate(window)
    except Exception as e:
        terminate(window, e)
