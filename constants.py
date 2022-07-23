"""Project wide constants related to data generation and training"""

CNN_LAYERS = 20
CNN_FILTERS = 64

TRAIN_PIXELS = CNN_LAYERS * 2 + 1  # width and height of training image
SCALING_FACTORS = (1, 2, 3, 4)
RANDOM_CROP_COUNT = 512
TRAIN_VALID_SPLIT = 0.6

EXPERIMENT_NAME = 'CNN, 41pixels, clipping, L2'

# dataset_41_B contains images with a multiple scales [1, 2, 3, 4]
DATASET_NAME = 'dataset_41_B'
