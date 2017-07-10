from easydict import EasyDict as edict
import numpy as np
from math import ceil

cfg = edict()

# Training process related constant
cfg.BATCH_SIZE = 5
cfg.EVAL_BATCH_SIZE = 5
cfg.MAX_STEPS = 20000
cfg.INITIAL_LEARNING_RATE = 0.001
cfg.MOVING_AVERAGE_DECAY = 0.9999
cfg.NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
cfg.LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# Network hyper-parameters
cfg.CONV_INIT_TYPE = 'orthogonal'
cfg.PRETRAINED_INITIALIZER = None
cfg.USE_LSTM = False
cfg.N_STATE = None
cfg.SEQUENCE_LENGTH = None

# Dataset dependent constant
cfg.DATASET_NANE = None
cfg.IMAGE_HEIGHT = None
cfg.IMAGE_WIDTH = None
cfg.IMAGE_DEPTH = None
cfg.NUM_CLASSES = None
cfg.NUM_TRAIN_EXAMPLES_PER_EPOCH = None
cfg.NUM_TEST_EXAMPLES_PER_EPOCH = None
cfg.NUM_VAL_EXAMPLES_PER_EPOCH = None
cfg.LOSS_WEIGHT = None
cfg.PARSE_LINE_FUNCTION = None
cfg.SET_ITER_FUNC = lambda N, b_size: int(ceil(N / float(b_size)))

# Queue dependent info
cfg.MIN_QUEUE_FRACTION = 0.4
