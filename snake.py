import pygame
import random
from qlearner import QLearning

from tools import Point
from collections import deque

from const import *



def run(learner, game_count):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake - Qvalues')
    clock = pygame.time.Clock()

    # Starting position of snake
    # if start_positionition is not defined, then initiate THE HEAD of a snake of length = 3
    # (so, the box will be within 2 (e.g: [0, 1, 2]) ->  box-3 (e.g: [box-3, box-2, box-1]))
    frames_since_last_apple = 0
    start_position          = None
    
    x = int(random.randrange(2*DOT_SIZE, SCREEN_WIDTH - 2*DOT_SIZE) / DOT_SIZE) * DOT_SIZE
    y = int(random.randrange(2*DOT_SIZE, SCREEN_HEIGHT - 2*DOT_SIZE) / DOT_SIZE) * DOT_SIZE
    start_position = Point(x, y)

    snake_list      = init_snake(start_position)
    length_of_snake = len(snake_list)

    x_head = snake_list[-1].x
    y_head = snake_list[-1].y

    # Create first apple
    applex = int(random.randrange(DOT_SIZE, SCREEN_WIDTH - DOT_SIZE) / DOT_SIZE) * DOT_SIZE
    appley = int(random.randrange(DOT_SIZE, SCREEN_HEIGHT - DOT_SIZE) / DOT_SIZE) * DOT_SIZE
    apple  = Point(applex, appley)

    dead = False
    reason = None
    while not dead:
        # Get action from agent
        action = learner.act(snake_list, apple)
        if action == 'left':
            x_head -= DOT_SIZE
        elif action == 'right':
            x_head += DOT_SIZE
        elif action == 'up':
            y_head -= DOT_SIZE
        elif action == 'down':
            y_head += DOT_SIZE

        # Move snake
        snake_head = Point(x_head,y_head)
        snake_list.append(snake_head)

        # Check if snake is off screen
        if (x_head >= SCREEN_WIDTH) or (x_head < 0) or (y_head >= SCREEN_HEIGHT) or (y_head < 0):
            reason = 'Screen'
            dead = True

        # Check if snake hit tail
        if snake_head in snake_list[:-1]:
            reason = 'Tail'
            dead = True

        # Check if snake ate apple
        if (x_head == apple.x) and (y_head == apple.y):
            applex = int(random.randrange(0, SCREEN_WIDTH - DOT_SIZE) / DOT_SIZE) * DOT_SIZE
            appley = int(random.randrange(0, SCREEN_HEIGHT - DOT_SIZE) / DOT_SIZE) * DOT_SIZE
            apple  = Point(applex, appley)
            length_of_snake += 1

        # Delete the last cell since we just added a head for moving, unless we ate a apple
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        # Draw apple, snake and update score
        screen.fill(GRAY)
    
        draw_apple(screen, apple)
        draw_snake(screen, snake_list)
        show_score(screen, length_of_snake - 3, game_count)
        pygame.display.update()

        # Update Q Table
        learner.update_qvalues(reason)
        
        # Next Frame
        clock.tick(FRAMESPEED)

    return (length_of_snake-3, reason)


def init_snake(start_position: set) -> list:
    '''
    Initialize the snake.
    starting_direction: ('u', 'd', 'l', 'r')
        direction that the snake should start facing. Whatever the direction is, the head
        of the snake will begin pointing that way.
    '''

    # initialize position of the head: ramdom in [2, self.board_size - 3]
    head = start_position

    possible_directions = ('up', 'down', 'left', 'right')
    starting_direction  = possible_directions[random.randint(0, 3)]

    # Body is below
    if starting_direction == 'up':
        snake = [Point(head.x, head.y + 2), Point(head.x, head.y + 1), head]
    # Body is above
    elif starting_direction == 'down':
        snake = [Point(head.x, head.y - 2), Point(head.x, head.y - 1), head]
    # Body is to the right
    elif starting_direction == 'left':
        snake = [Point(head.x + 2, head.y), Point(head.x + 1, head.y), head]
    # Body is to the left
    elif starting_direction == 'right':
        snake = [Point(head.x - 2, head.y), Point(head.x - 1, head.y), head]

    # snake_array    = deque(snake)
    # body_locations = set(snake)
    return snake



def draw_apple(screen, apple):
    pygame.draw.rect(screen, GREEN, [apple.x, apple.y, DOT_SIZE, DOT_SIZE])   

def show_score(screen, score, game_count):
    font = pygame.font.SysFont('comicsansms', 18)
    value = font.render(f'Game: {game_count}, Score: {score}', True, RED)
    screen.blit(value, [0, 0])

def draw_snake(screen, snake_list):
    for p in snake_list[:-1]:
        pygame.draw.rect(screen, BLUE, [p.x, p.y, DOT_SIZE, DOT_SIZE])

    pygame.draw.rect(screen, RED, [snake_list[-1].x, snake_list[-1].y, DOT_SIZE, DOT_SIZE])



def main():
    pygame.init()

    game_count = 1

    learner = QLearning(SCREEN_WIDTH, SCREEN_HEIGHT, DOT_SIZE)

    while True:
        learner.reset()
        if game_count > QVALUES_N:
            learner.epsilon = 0.
        else:
            learner.epsilon = 0.1
        
        score, reason = run(learner, game_count)
        
        game_count += 1
        if game_count % QVALUES_N == 0: # Save qvalues every QVALUES_N games
            print('Save Q-values to JSON file in [data/qvalues.json]... ')
            learner.save_qvalues()



if __name__ == '__main__':
    main()

