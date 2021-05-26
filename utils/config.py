TITLE = 'Deep Q-Learning Snake'
BG_COLOR   = 'gray'

CELL_SHAPE = 'circle'
SNAKE_COLOR = 'red'
SNAKE_START_LOC_H = 0
SNAKE_START_LOC_V = 0

APPLE_SHAPE = 'circle'
APPLE_COLOR = 'green'

TEXT_ALIGN = 'left'
TEXT_FONT  = 'Courier'
TEXT_SIZE  = 10


N_STEPS         = 30           # number of steps from top-to-bottom/left-to-right  of screen
PIXELS_PER_STEP = 20
MARGINS         = 2*PIXELS_PER_STEP
BOARD_HEIGHT    = PIXELS_PER_STEP*N_STEPS
BOARD_WIDTH     = PIXELS_PER_STEP*N_STEPS
WIN_HEIGHT      = PIXELS_PER_STEP*N_STEPS + PIXELS_PER_STEP   # pixel height + border on both sides
WIN_WIDTH       = PIXELS_PER_STEP*N_STEPS + PIXELS_PER_STEP   # pixel width + border on both sides

SLEEP = 0.1     # time to wait between steps



PAUSE = 0
UP    = 1
DOWN  = 2
LEFT  = 3
RIGHT = 4


OUT_OF_WINDOW = 2*WIN_HEIGHT


# For greedy epsilon policy
EPSILON       = 1.
EPSILON_MIN   = .01
EPSILON_DECAY = .995

DISCOUNT_RATE = .95
BATCH_SIZE    = 64
LEARNING_RATE = 1.e-4
LAYER_UNITS   = [128, 128, 228]