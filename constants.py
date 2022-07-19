"""Project wide constants related to data generation and training"""

CNN_LAYERS = 20
CNN_FILTERS = 64

TRAIN_PIXELS = CNN_LAYERS * 2 + 1  # width and height of training image
RANDOM_CROP_COUNT = 1024
TRAIN_VALID_SPLIT = 0.6
DATASET_NAME = 'dataset_41_A'
