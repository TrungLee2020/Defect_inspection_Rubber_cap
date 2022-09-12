from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D


def build_siamese_model(inputShape, embeddingDim=48):
    # embeddingDim: Output dimensionality of the final fully-connected layer in the network.
    inputs = Input(inputShape)

    # define first set of CONV => RELU => POOL => DROPOUT layer
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Sencond layer
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # prepare final output
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)

    # Build Model
    model = Model(inputs, outputs)

    return model
