import random
from collections import defaultdict
from enum import Enum

import numpy as np
from kivy import properties as kp
from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import sp
from kivy.uix.label import Label
from kivy.uix.widget import Widget

WINDOW_SIZE = [600, 600]
SPRITE_SIZE = sp(20)
COLS = int(WINDOW_SIZE[0] * 2 / SPRITE_SIZE)
ROWS = int(WINDOW_SIZE[1] * 2 / SPRITE_SIZE)

ALPHA = .5


class Direction(Enum):
    """Directions:
• LEFT(0)
• UP (1)
• RIGHT (2)
• DOWN (3)"""
    LEFT, UP, RIGHT, DOWN = range(4)


class RunState(Enum):
    PAUSED, LOOPING = range(2)


direction_vector_map = {
    Direction.LEFT: [-1, 0],
    Direction.UP: [0, 1],
    Direction.RIGHT: [1, 0],
    Direction.DOWN: [0, -1]
}

direction_group = {
    Direction.LEFT: 'horizontal',
    Direction.UP: 'vertical',
    Direction.RIGHT: 'horizontal',
    Direction.DOWN: 'vertical'
}


class Block(Widget):
    coord = kp.ListProperty([0, 0])
    bgcolor = kp.ListProperty([0, 0, 0, 0])


class SnakeBlock(Block):
    pass


# Dictionary of sprites. By using defaultdict it is possible to specify what
# to do if a sprite is not already present (in this case, create it).
SPRITES = defaultdict(lambda: SnakeBlock())


class Food(Block):
    pass


class Score(Label):
    coord = kp.ListProperty([4, COLS - 3])


class Snake(App):
    movespeed = .05
    initial_lenght = 6 - 1

    sprite_size = kp.NumericProperty(SPRITE_SIZE)

    Window.size = WINDOW_SIZE

    head = kp.ListProperty([0, 0])
    snake = kp.ListProperty()
    lenght = kp.NumericProperty(initial_lenght)

    score = kp.NumericProperty(0)
    deaths = kp.NumericProperty(0)
    steps = kp.NumericProperty(0)

    food = kp.ListProperty([0, 0])
    food_sprite = kp.ObjectProperty(Food)

    direction = kp.ObjectProperty(random.choice(list(Direction)), options=(Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN))
    buffer_direction = kp.ObjectProperty(random.choice(list(Direction)), allownone=True, options=(Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN))
    block_input = kp.BooleanProperty(False)

    alpha = kp.NumericProperty(0)

    cols = COLS
    rows = ROWS

    def on_start(self):
        print("Cols: {}, Rows: {}".format(COLS, ROWS))

        self.keyboard = Window.request_keyboard(self.on_keyboard_close, self.root)
        self.keyboard.bind(on_key_down=self.key_handler)
        Window.bind(on_resize=self.on_resize)

        self.head = self.new_head_location
        self.food_sprite = Food()
        self.food = self.new_food_location

        self.stats_label = Score(markup=True, font_size='20sp')
        self.on_score()

        self.scheduled_functions = []
        self.scheduled_events = []

        self.run_state = RunState.PAUSED

        self.done = False
        self.reward = 0

    def loop(self):
        if self.run_state == RunState.PAUSED:
            self.run_state = RunState.LOOPING
            # move
            self.scheduled_events.append(Clock.schedule_interval(self.move, self.movespeed))
            self.scheduled_functions.append(self.move)

    def pause(self):
        if self.run_state == RunState.LOOPING:
            self.run_state = RunState.PAUSED
            for event in self.scheduled_events.copy():
                Clock.unschedule(event)
                self.scheduled_events.remove(event)

    def step(self, action=None):
        if self.run_state == RunState.PAUSED:
            self.set_direction(action if action else self.generate_random_direction())
            return self.move()

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

    def set_move_speed(self, speed):
        # unschedule all the previously scheduled events
        for event in self.scheduled_events:
            Clock.unschedule(event)
            self.scheduled_events.remove(event)

        # update speed
        self.movespeed = speed

        # reschedule all the events with the updated speed
        for func in self.scheduled_functions:
            self.scheduled_events.append(Clock.schedule_interval(func, self.movespeed))

    def update_stats_label(self):
        self.stats_label.text = "[b]Score: " + str(self.score) + "\nDeaths: " + str(self.deaths) + "\nSteps: " + str(self.steps) + "[/b]"

    # called every time the window is resized
    def on_resize(self, *args):
        Window.size = WINDOW_SIZE

    def on_keyboard_close(self):
        self.keyboard.unbind(on_key_down=self.key_handler)
        self.keyboard = None

    def key_direction_mapping(self, key):
        if key == 'left':
            return Direction.LEFT
        elif key == 'up':
            return Direction.UP
        elif key == 'right':
            return Direction.RIGHT
        elif key == 'down':
            return Direction.DOWN
        else:
            raise KeyError

    def key_handler(self, *args):
        try:
            key = args[1][1]
            if self.valid_movement_key(key):
                self.set_direction(self.key_direction_mapping(key))
            elif key == 'n':
                self.set_move_speed(self.movespeed / 4)
            elif key == 'm':
                self.set_move_speed(self.movespeed * 4)

            elif key == 'j':
                self.step(self.direction)
            elif key == 'k':
                self.loop()
            elif key == 'l':
                self.pause()

            else:
                raise KeyError
        except KeyError:
            pass

    def valid_movement_key(self, key):
        """Check if given parameter is a valid movement key. \nPossible values are:\n- left\n- up\n- right\n- down."""
        if key not in ['left', 'up', 'right', 'down']:
            return False
        return True

    def set_direction(self, new_direction):
        """Try to change the direction of the snake. In order for the direction to be changed, the given direction must be orthogonal with respect to the current one. If the direction has already been changed for the current frame, the given direction is buffered and scheduled for the next frame."""
        if direction_group[new_direction] != direction_group[self.direction]:
            if self.block_input:
                self.buffer_direction = new_direction
            else:
                self.direction = new_direction
                self.block_input = True
                # print("Direction: ", self.direction)

    def generate_random_direction(self, *args):
        return random.choice(list(Direction))

    def set_random_direction(self, *args):
        self.set_direction(self.generate_random_direction())

    @property
    def direction_vector(self):
        return direction_vector_map[self.direction]

    def move(self, *args):
        # release lock on nn action
        self.done = False
        self.reward = 0

        # release input block
        self.block_input = False

        # calculate new head coords
        new_head = [sum(x) for x in zip(self.head, direction_vector_map[self.direction])]

        # check for collisions
        if not self.check_in_bounds(new_head) or new_head in self.snake:
            self.done = True
            self.reward = -1
            self.die()
        else:
            # check for food eaten
            if new_head == self.food:
                self.lenght += 1
                self.score += 1
                self.reward = 1
                self.food = self.new_food_location

            # check if a direction was previously buffered
            if self.buffer_direction:
                self.set_direction(self.buffer_direction)
                self.buffer_direction = None

            self.head = new_head

        self.steps += 1

        return self.generate_observation()

    def on_score(self, *args):
        """Called every time the `score` field changes"""
        self.update_stats_label()
        if not self.stats_label.parent:
            self.root.add_widget(self.stats_label)

    def on_steps(self, *args):
        """Called every time the `steps` field changes"""
        self.on_score(args)

    def on_food(self, *args):
        """Called every time the `food` field changes."""
        self.food_sprite.coord = self.food
        if not self.food_sprite.parent:
            self.root.add_widget((self.food_sprite))

    def on_head(self, *args):
        """Called every time the `head` field changes."""
        self.snake = self.snake[-self.lenght:] + [self.head]

    def on_snake(self, *args):
        """Called every time the `snake` field changes."""
        for index, coord in enumerate(self.snake):
            sprite = SPRITES[index]
            sprite.coord = coord
            if not sprite.parent:
                self.root.add_widget(sprite)

    @property
    def new_head_location(self):
        # the '2' is beacuse the head cannot be within the borders
        return [random.randint(2, dim - 2) for dim in [COLS, ROWS]]

    @property
    def new_food_location(self):
        while True:
            # generate new coords for food until they are valid (i.e. they don't overlap with a snake block)
            food = [random.randint(0, dim - 1) for dim in [COLS, ROWS]]
            if food not in self.snake and food != self.food:
                return food

    def check_in_bounds(self, pos):
        """Check if the given position is inside the boundaries."""
        return all(0 <= pos[x] < dim for x, dim in enumerate([COLS, ROWS]))

    def die(self):
        self.deaths += 1
        self.restart()

    def restart(self):
        # clear all the widgets of the canvas
        self.root.clear_widgets()

        # red animation for when the snake dies
        self.alpha = ALPHA
        Animation(alpha=0, duration=self.movespeed).start(self)

        # remove all the snake blocks
        self.snake.clear()
        self.lenght = self.initial_lenght

        # reset score
        self.score = 0
        self.steps = 0
        self.on_score()

        # generates new blocks
        self.head = self.new_head_location
        self.food = self.new_food_location

        # # the snake will now start moving in a new random direction
        # self.direction = random.choice(list(Direction))

        return self.generate_observation()


if __name__ == "__main__":
    print("")
    print("Keys:")
    print("- 'j' to perform a single step")
    print("- 'k' for continuous play")
    print("- 'l' to stop continuous play")
    print("- 'n' to increase game speed")
    print("- 'm' to decrease game speed")
    print("")

    try:
        Snake().run()
    except KeyboardInterrupt:
        Snake().stop()
        raise KeyboardInterrupt
