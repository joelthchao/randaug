from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model


def build_simple_model(input_shape, num_classes):
    input = Input(input_shape)
    net = Conv2D(96, (3, 3), padding='same', activation='relu') (input)
    net = Conv2D(96, (3, 3), padding='same', activation='relu') (net)
    net = MaxPooling2D((3, 3), strides=2)(net)

    net = Conv2D(192, (3, 3), padding='same', activation='relu') (net)
    net = Conv2D(192, (3, 3), padding='same', activation='relu') (net)
    net = MaxPooling2D((3, 3), strides=2)(net)
    flatten = Flatten() (net)

    net = Dense(512, activation='relu') (flatten)
    out = Dense(num_classes, activation='softmax') (net)

    return Model(input, out)
