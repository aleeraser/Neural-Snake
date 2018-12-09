import random
from collections import defaultdict
from enum import Enum

from kivy import properties as kp
from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import sp
from kivy.uix.label import Label
from kivy.uix.widget import Widget

WINDOW_SIZE = [800, 600]
SPRITE_SIZE = sp(20)
COLS = int(Window.width / SPRITE_SIZE)
ROWS = int(Window.height / SPRITE_SIZE)

ALPHA = .5

class Direction(Enum):
    LEFT, RIGHT, UP, DOWN = range(4)


direction_values = {
    Direction.LEFT: [-1, 0],
    Direction.RIGHT: [1, 0],
    Direction.UP: [0, 1],
    Direction.DOWN: [0, -1]
}

direction_group = {
    Direction.LEFT: 'horizontal',
    Direction.RIGHT: 'horizontal',
    Direction.UP: 'vertical',
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
    coord = kp.ListProperty([2, ROWS - 2])


class Snake(App):
    movespeed = .05
    initial_lenght = 4

    sprite_size = kp.NumericProperty(SPRITE_SIZE)

    head = kp.ListProperty([0, 0])
    snake = kp.ListProperty()
    lenght = kp.NumericProperty(initial_lenght)

    score = kp.NumericProperty(0)
    deaths = kp.NumericProperty(0)

    food = kp.ListProperty([0, 0])
    food_sprite = kp.ObjectProperty(Food)

    direction = kp.ObjectProperty(random.choice(list(Direction)), options=(Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN))
    buffer_direction = kp.ObjectProperty(random.choice(list(Direction)), allownone=True, options=(Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN))
    block_input = kp.BooleanProperty(False)

    alpha = kp.NumericProperty(0)

    def on_start(self):
        Window.size = WINDOW_SIZE

        self.keyboard = Window.request_keyboard(self.on_keyboard_close, self.root)
        self.keyboard.bind(on_key_down=self.key_handler)
        Window.bind(on_resize=self.on_resize)

        self.head = self.new_head_location
        self.food_sprite = Food()
        self.food = self.new_food_location

        self.scoreLabel = Score(markup=True, font_size='20sp')
        self.on_score()

        self.scheduledFunctions = []
        self.scheduledEvents = []
        # NN decision
        # self.scheduledEvents.append(Clock.schedule_interval(self.NNdecision, self.movespeed))
        # self.scheduledFunctions.append(self.NNdecision)
        # move
        self.scheduledEvents.append(Clock.schedule_interval(self.move, self.movespeed))
        self.scheduledFunctions.append(self.move)

    def setMoveSpeed(self, speed):
        for event in self.scheduledEvents:
            Clock.unschedule(event)
            self.scheduledEvents.remove(event)

        self.movespeed = speed

        for func in self.scheduledFunctions:
            self.scheduledEvents.append(Clock.schedule_interval(func, self.movespeed))

    def updateScoreLabel(self):
        self.scoreLabel.text = "[b]Score: " + str(self.score) + "\nDeaths: " + str(self.deaths) + "[/b]"

    def on_resize(self, *args):
        Window.size = WINDOW_SIZE

    def on_keyboard_close(self):
        self.keyboard.unbind(on_key_down=self.key_handler)
        self.keyboard = None

    def key_direction_mapping(self, key):
        if key == 'left':
            return Direction.LEFT
        elif key == 'right':
            return Direction.RIGHT
        elif key == 'up':
            return Direction.UP
        elif key == 'down':
            return Direction.DOWN
        else:
            raise KeyError

    def key_handler(self, *args):
        try:
            key = args[1][1]
            if self.valid_key(key):
                self.try_change_direction(self.key_direction_mapping(key))
            elif key == 'n':
                self.setMoveSpeed(self.movespeed / 4)
            elif key == 'm':
                self.setMoveSpeed(self.movespeed * 4)
            else:
                raise KeyError
        except KeyError:
            pass

    def valid_key(self, key):
        if key not in ['left', 'up', 'right', 'down']:
            return False
        return True

    def try_change_direction(self, new_direction):
        if direction_group[new_direction] != direction_group[self.direction]:
            if self.block_input:
                self.buffer_direction = new_direction
            else:
                self.direction = new_direction
                self.block_input = True
                # print("Direction: ", self.direction)

    def NNdecision(self, *args):
        self.try_change_direction(random.choice(list(Direction)))

    def move(self, *args):
        self.block_input = False
        new_head = [sum(x) for x in zip(self.head, direction_values[self.direction])]
        if not self.check_in_bounds(new_head) or new_head in self.snake:
            return self.die()
        if new_head == self.food:
            self.lenght += 1
            self.score += 1

            self.food = self.new_food_location
        if self.buffer_direction:
            self.try_change_direction(self.buffer_direction)
            self.buffer_direction = None
        self.head = new_head

    # called every time the score changes
    def on_score(self, *args):
        self.updateScoreLabel()
        if not self.scoreLabel.parent:
            self.root.add_widget(self.scoreLabel)

    # called every time the food changes
    def on_food(self, *args):
        self.food_sprite.coord = self.food
        if not self.food_sprite.parent:
            self.root.add_widget((self.food_sprite))

    # called every time the head changes
    def on_head(self, *args):
        self.snake = self.snake[-self.lenght:] + [self.head]

    # called every time the snake changes
    def on_snake(self, *args):
        for index, coord in enumerate(self.snake):
            sprite = SPRITES[index]
            sprite.coord = coord
            if not sprite.parent:
                self.root.add_widget(sprite)

    @property
    def new_head_location(self):
        return [random.randint(2, dim - 2) for dim in [COLS, ROWS]]

    @property
    def new_food_location(self):
        while True:
            food = [random.randint(0, dim - 1) for dim in [COLS, ROWS]]
            if food not in self.snake and food != self.food:
                return food

    def check_in_bounds(self, pos):
        return all(0 <= pos[x] < dim for x, dim in enumerate([COLS, ROWS]))

    def die(self):
        self.root.clear_widgets()

        # red animation for when the snake dies
        self.alpha = ALPHA
        Animation(alpha=0, duration=self.movespeed).start(self)

        self.snake.clear()
        self.lenght = self.initial_lenght

        self.score = 0
        self.deaths += 1
        self.on_score()

        self.head = self.new_head_location
        self.food = self.new_food_location

        self.direction = random.choice(list(Direction))


if __name__ == "__main__":
    Snake().run()
