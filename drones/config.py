import sys
sys.dont_write_bytecode = True

A_MAT = [[0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
B_MAT = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]

K_MAT = [[1, 0, 0, 3, 0, 0, 3, 0],
         [0, 1, 0, 0, 3, 0, 0, 3],
         [0, 0, 1, 0, 0, 3, 0, 0]]

TIME_STEP = 0.05
TIME_STEP_EVAL = 0.05
TOP_K = 8
OBS_RADIUS = 1.0

DIST_MIN_THRES = 0.8
DIST_MIN_CHECK = 0.6
DIST_SAFE = 1.0
DIST_TOLERATE = 0.7

ALPHA_CBF = 1.0
WEIGHT_DECAY = 1e-8

TRAIN_STEPS = 70000
EVALUATE_STEPS = 5
INNER_LOOPS = 40
INNER_LOOPS_EVAL = 40
DISPLAY_STEPS = 10
SAVE_STEPS = 200

LEARNING_RATE = 1e-4
REFINE_LEARNING_RATE = 1.0
REFINE_LOOPS = 40
