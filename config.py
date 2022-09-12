import os

IMG_SHAPE = (28, 28, 1)

BATCH_SIZE = 16
EPOCHS = 20

BASE_OUTPUT = "./output"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])