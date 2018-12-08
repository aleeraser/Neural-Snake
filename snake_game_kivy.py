from collections import defaultdict
from random import randint

from kivy import properties as kp
from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import sp
from kivy.uix.widget import Widget

WINDOW_SIZE = [800, 600]
SPRITE_SIZE = sp(20)
COLS = int(Window.width / SPRITE_SIZE)
ROWS = int(Window.height / SPRITE_SIZE)

LENGHT = 4
MOVESPEED = .05

ALPHA = .5

LEFT = 'left'
UP = 'up'
RIGHT = 'right'
DOWN = 'down'

direction_values = {LEFT: [-1, 0],
                    UP: [0, 1],
                    RIGHT: [1, 0],
                    DOWN: [0, -1]}

direction_group = {LEFT: 'horizontal',
                   UP: 'vertical',
                   RIGHT: 'horizontal',
                   DOWN: 'vertical', }


class Block(Widget):
    coord = kp.ListProperty([0, 0])
    bgcolor = kp.ListProperty([0, 0, 0, 0])


class SnakeBlock(Block):
    pass


# Dictionary of sprites. By using defaultdict we specify what to do if a sprite
# is not already present (in this case, we create it).
SPRITES = defaultdict(lambda: SnakeBlock())


class Food(Block):
    pass


class Snake(App):
    sprite_size = kp.NumericProperty(SPRITE_SIZE)

    head = kp.ListProperty([0, 0])
    snake = kp.ListProperty()
    lenght = kp.NumericProperty(LENGHT)

    food = kp.ListProperty([0, 0])
    food_sprite = kp.ObjectProperty(Food)

    direction = kp.StringProperty(RIGHT, options=(LEFT, UP, RIGHT, DOWN))
    buffer_direction = kp.StringProperty(RIGHT, allownone=True, options=(LEFT, UP, RIGHT, DOWN))
    block_input = kp.BooleanProperty(False)

    alpha = kp.NumericProperty(0)

    wallsEnabled = True

    def on_start(self):
        Window.size = WINDOW_SIZE

        self.keyboard = Window.request_keyboard(self.on_keyboard_close, self.root)
        self.keyboard.bind(on_key_down=self.key_handler)
        Window.bind(on_resize=self.on_resize)

        self.head = self.new_head_location
        self.food_sprite = Food()
        self.food = self.new_food_location
        Clock.schedule_interval(self.move, MOVESPEED)

        self.root.add_widget()

    def on_resize(self, *args):
        Window.size = WINDOW_SIZE
        self.die()

    def on_keyboard_close(self):
        self.keyboard.unbind(on_key_down=self.key_handler)
        self.keyboard = None

    def key_handler(self, *args):
        try:
            key = args[1][1]
            if self.valid_key(key):
                self.try_change_direction(key)
            else:
                raise KeyError
        except KeyError:
            pass

    def valid_key(self, key):
        if key not in [RIGHT, LEFT, UP, DOWN]:
            return False
        return True

    def try_change_direction(self, new_direction):
        if direction_group[new_direction] != direction_group[self.direction]:
            if self.block_input:
                self.buffer_direction = new_direction
            else:
                self.direction = new_direction
                self.block_input = True

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
        return [randint(2, dim - 2) for dim in [COLS, ROWS]]

    @property
    def new_food_location(self):
        while True:
            food = [randint(0, dim - 1) for dim in [COLS, ROWS]]
            if food not in self.snake and food != self.food:
                return food

    def move(self, *args):
        self.block_input = False
        new_head = [sum(x) for x in zip(self.head, direction_values[self.direction])]
        if not self.check_in_bounds(new_head) or new_head in self.snake:
            return self.die()
        if new_head == self.food:
            self.lenght += 1
            self.food = self.new_food_location
        if self.buffer_direction:
            self.try_change_direction(self.buffer_direction)
            self.buffer_direction = None
        self.head = new_head

    def check_in_bounds(self, pos):
        return all(0 <= pos[x] < dim for x, dim in enumerate([COLS, ROWS]))

    def die(self):
        self.root.clear_widgets()

        # red animation for when the snake dies
        self.alpha = ALPHA
        Animation(alpha=0, duration=MOVESPEED).start(self)

        self.snake.clear()
        self.lenght = LENGHT
        self.head = self.new_head_location
        self.food = self.new_food_location


if __name__ == "__main__":
    Snake().run()
