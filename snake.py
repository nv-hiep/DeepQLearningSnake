import turtle
import random
import time
import math
import gym
from gym       import spaces
from gym.utils import seeding

from utils.config import *

class Snake(gym.Env):

    def __init__(self,
                 human = False,
                 env_info = {'state_space':None}):
        super(Snake, self).__init__()

        self.seed()

        self.action_space = 4    # 4 possible actions: up, down, left, right
        self.state_space  = 12
        self.dead         = False
        self.reward       = 0
        self.total_score  = 0
        self.maximum      = 0
        self.human        = human
        self.env_info     = env_info

        ## GAME CREATION WITH TURTLE (RENDER?)
        # screen/background
        self.win = turtle.Screen()
        self.win.title(TITLE)
        self.win.bgcolor(BG_COLOR)
        self.win.tracer(0)
        self.win.setup(width=WIN_WIDTH, height=WIN_HEIGHT)
                
        # snake
        self.init_snake()

        # apple
        self.init_apple()

        # distance between apple and snake
        self.dist = math.sqrt((self.snake.xcor()-self.apple.xcor())**2 + (self.snake.ycor()-self.apple.ycor())**2)

        # score
        self.init_score()

        # control
        self.win.listen()
        self.win.onkey(self.go_up, 'Up')
        self.win.onkey(self.go_right, 'Right')
        self.win.onkey(self.go_down, 'Down')
        self.win.onkey(self.go_left, 'Left')

    
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def init_snake(self) -> None:
        self.snake = turtle.Turtle()
        self.snake.shape(CELL_SHAPE)
        self.snake.speed(0)
        self.snake.penup()
        self.snake.color(SNAKE_COLOR)

        head_x = random.randint(-N_STEPS/2+3, N_STEPS/2-3)*PIXELS_PER_STEP
        head_y = random.randint(-N_STEPS/2+3, N_STEPS/2-3)*PIXELS_PER_STEP

        self.snake.goto(round(head_x), round(head_y))
        self.snake.direction = PAUSE

        # snake body, add first element (for location of snake's head)
        self.snake_body = []
        self.add_cell()
        self.snake_body[0].goto(round(head_x), round(head_y)-0*PIXELS_PER_STEP)
        self.add_cell()
        self.snake_body[1].goto(round(head_x), round(head_y)-1*PIXELS_PER_STEP)
        self.add_cell()
        self.snake_body[2].goto(round(head_x), round(head_y)-2*PIXELS_PER_STEP)

    
    def move_head(self):
        '''
        Note: self.snake.xcor() from -BOARD_WIDTH/2 to BOARD_WIDTH/2
              self.snake.ycor() from -BOARD_HEIGHT/2 to BOARD_HEIGHT/2
        '''
        if self.snake.direction == PAUSE:
            self.reward = 0
        if self.snake.direction == UP:
            y = self.snake.ycor()
            self.snake.sety(y + PIXELS_PER_STEP)
        if self.snake.direction == RIGHT:
            x = self.snake.xcor()
            self.snake.setx(x + PIXELS_PER_STEP)
        if self.snake.direction == DOWN:
            y = self.snake.ycor()
            self.snake.sety(y - PIXELS_PER_STEP)
        if self.snake.direction == LEFT:
            x = self.snake.xcor()
            self.snake.setx(x - PIXELS_PER_STEP)
        
    
    def go_up(self):
        if self.snake.direction != DOWN:
            self.snake.direction = UP
    
    
    def go_down(self):
        if (self.snake.direction != UP) & (self.snake.direction != PAUSE):
            self.snake.direction = DOWN
    
    
    def go_right(self):
        if self.snake.direction != LEFT:
            self.snake.direction = RIGHT
    
    
    def go_left(self):
        if self.snake.direction != RIGHT:
            self.snake.direction = LEFT

    

    #
    # APPLE
    #
    def random_coordinates(self):
        apple_x = random.randint(-N_STEPS/2, N_STEPS/2)
        apple_y = random.randint(-N_STEPS/2, N_STEPS/2)
        return apple_x, apple_y


    def generate_apple(self):
        count = 0
        while True:
            self.apple.x, self.apple.y = self.random_coordinates()
            self.apple.goto(round(self.apple.x*PIXELS_PER_STEP), round(self.apple.y*PIXELS_PER_STEP))

            # Check whether the apple is not generated in snake's body
            if not self.eat_apple():
                return True
            if count > BOARD_HEIGHT*BOARD_WIDTH:
                print('You win!')
                exit(0)
                return False

    def init_apple(self):
        self.apple = turtle.Turtle()
        self.apple.speed(0)
        self.apple.shape(APPLE_SHAPE)
        self.apple.color(APPLE_COLOR)
        self.apple.penup()

        self.generate_apple()
    
    def move_apple(self):
        if self.snake.distance(self.apple) < PIXELS_PER_STEP:
            self.generate_apple()
            self.update_score()
            self.add_cell()
            return True



    #
    # SCORES
    #
    def init_score(self):
        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color('black')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(-BOARD_WIDTH/2, BOARD_HEIGHT/2-PIXELS_PER_STEP)
        self.score.write(f'Score: {self.total_score}   Best Score: {self.maximum}   Reward : {self.reward}',
                         align=TEXT_ALIGN,
                         font=(TEXT_FONT, TEXT_SIZE, 'normal'))



    def update_score(self):
        self.total_score += 1
        if self.total_score >= self.maximum:
            self.maximum = self.total_score
        self.score.clear()
        self.score.write(f'Score: {self.total_score}   Best Score: {self.maximum}   Reward : {self.reward}',
                         align=TEXT_ALIGN,
                         font=(TEXT_FONT, TEXT_SIZE, 'normal'))


    def reset_score(self):
        self.score.clear()
        self.total_score = 0
        self.score.write(f'Score: {self.total_score}   Best Score: {self.maximum}   Reward : {self.reward}',
                         align=TEXT_ALIGN,
                         font=(TEXT_FONT, TEXT_SIZE, 'normal'))
                    



    #
    # SNAKE's BODY
    #
    def add_cell(self):
        body = turtle.Turtle()
        body.speed(0)
        body.shape(CELL_SHAPE)
        body.color(SNAKE_COLOR)
        body.penup()
        self.snake_body.append(body)
        

    def move_body(self):
        length = len(self.snake_body)
        if length > 0:
            for index in range(length-1, 0, -1):
                x = self.snake_body[index-1].xcor()
                y = self.snake_body[index-1].ycor()
                self.snake_body[index].goto(x, y)

            self.snake_body[0].goto(self.snake.xcor(), self.snake.ycor())
        
    
    def distance_to_apple(self):
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.snake.xcor()-self.apple.xcor())**2 + (self.snake.ycor()-self.apple.ycor())**2)


    def hit_self(self):
        if len(self.snake_body) > 1:
            for body in self.snake_body[1:]:
                if body.distance(self.snake) < PIXELS_PER_STEP:
                    self.reset_score()
                    return True     

    def eat_apple(self):
        '''
        Check if the apple is generated in snake's body
        '''
        if len(self.snake_body) > 0:
            for body in self.snake_body:
                if body.distance(self.apple) < PIXELS_PER_STEP:
                    return True
        return False

    def hit_wall(self):
        if (self.snake.xcor() > BOARD_WIDTH/2 ) or (self.snake.xcor() < -BOARD_WIDTH/2) or\
           (self.snake.ycor() > BOARD_HEIGHT/2) or (self.snake.ycor() < -BOARD_HEIGHT/2):
            self.reset_score()
            return True
    
    def reset(self):
        # turtle.clear()
        # turtle.reset()

        if self.human:
            time.sleep(1)

        # Move the apple and snake out of the window
        # to clear the screen
        self.apple.goto(OUT_OF_WINDOW, OUT_OF_WINDOW)
        self.snake.goto(OUT_OF_WINDOW, OUT_OF_WINDOW)
        for body in self.snake_body:
            body.goto(OUT_OF_WINDOW, OUT_OF_WINDOW)
        self.snake_body = []

        # Initiate the apple and snake
        
        self.init_snake()
        self.init_apple()

        self.reward       = 0
        self.total_score  = 0
        self.dead         = False

        return self.get_state()



    #
    # play
    #
    def play(self):
        reward_given = False
        self.win.update()
        self.move_head()
        
        # If snake eats an apple, reward 10 points
        if self.move_apple():
            self.reward = 10
            reward_given = True
        
        if self.snake.direction != PAUSE:
            self.move_body()
        
        self.distance_to_apple()
            
        if self.hit_self():
            self.reward  = -100
            reward_given = True
            self.dead    = True
            if self.human:
                self.reset()
        
        if self.hit_wall():
            self.reward  = -100
            reward_given = True
            self.dead    = True
            if self.human:
                self.reset()
        
        if not reward_given:
            self.reward = 1 if (self.dist < self.prev_dist) else -1

        if self.human:
            time.sleep(SLEEP)
            state = self.get_state()





    #
    # AI agent
    #
    def step(self, action):
        if action == 0:
            self.go_up()
        if action == 1:
            self.go_right()
        if action == 2:
            self.go_down()
        if action == 3:
            self.go_left()
        self.play()
        state = self.get_state()
        return state, self.reward, self.dead, {} #  state, rewards, dead, infor


    def get_state(self):
        # snake coordinates abs
        self.snake.x, self.snake.y = self.snake.xcor()/N_STEPS, self.snake.ycor()/N_STEPS   
        # snake coordinates scaled 0-1
        self.snake.xsc, self.snake.ysc = self.snake.x/N_STEPS+0.5, self.snake.y/N_STEPS+0.5
        # apple coordintes scaled 0-1 
        self.apple.xsc, self.apple.ysc = self.apple.x/N_STEPS+0.5, self.apple.y/N_STEPS+0.5

        # wall check
        if self.snake.y >= N_STEPS/2:
            wall_up   = 1
            wall_down = 0
        elif self.snake.y <= -N_STEPS/2:
            wall_up   = 0
            wall_down = 1
        else:
            wall_up   = 0
            wall_down = 0
        if self.snake.x >= N_STEPS/2:
            wall_right = 1
            wall_left  = 0
        elif self.snake.x <= -N_STEPS/2:
            wall_right = 0
            wall_left  = 1
        else:
            wall_right = 0
            wall_left  = 0

        # body close
        body_up = []
        body_right = []
        body_down = []
        body_left = []
        if len(self.snake_body) > 3:
            for body in self.snake_body[3:]:
                if body.distance(self.snake) == PIXELS_PER_STEP:
                    if body.ycor() < self.snake.ycor():
                        body_down.append(1)
                    elif body.ycor() > self.snake.ycor():
                        body_up.append(1)
                    if body.xcor() < self.snake.xcor():
                        body_left.append(1)
                    elif body.xcor() > self.snake.xcor():
                        body_right.append(1)
        
        body_up    = 1 if len(body_up)    > 0 else 0
        body_right = 1 if len(body_right) > 0 else 0
        body_down  = 1 if len(body_down)  > 0 else 0
        body_left  = 1 if len(body_left)  > 0 else 0

        # state: apple_up, apple_right, apple_down, apple_left, obstacle_up, obstacle_right, obstacle_down, obstacle_left, direction_up, direction_right, direction_down, direction_left
        if self.env_info['state_space'] == 'coordinates':
            return [self.apple.xsc, self.apple.ysc, self.snake.xsc, self.snake.ysc, \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    int(self.snake.direction == UP), int(self.snake.direction == RIGHT), int(self.snake.direction == DOWN), int(self.snake.direction == LEFT)]
        elif self.env_info['state_space'] == 'no direction':
            return [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    0, 0, 0, 0]
        elif self.env_info['state_space'] == 'no body knowledge':
            return [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    wall_up, wall_right, wall_down, wall_left, \
                    int(self.snake.direction == UP), int(self.snake.direction == RIGHT), int(self.snake.direction == DOWN), int(self.snake.direction == LEFT)]
        else:
            return [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    int(self.snake.direction == UP), int(self.snake.direction == RIGHT), int(self.snake.direction == DOWN), int(self.snake.direction == LEFT)]

    # def bye(self):
    #     self.win.bye()



if __name__ == '__main__':            
    human = True
    env = Snake(human=human)

    # Enjoy playing yourself
    if human:
        while True:
            env.play()