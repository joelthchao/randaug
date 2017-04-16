from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model


def build_simple_model(input_shape, num_classes):
    input = Input(input_shape)
    net = Conv2D(96, (3, 3), padding='same', activation='relu') (input)
    net = Conv2D(96, (3, 3), padding='same', activation='relu') (net)
    net = MaxPooling2D((3, 3), strides=2)(net)

    net = Conv2D(192, (3, 3), padding='same', activation='relu') (net)
    net = Conv2D(192, (3, 3), padding='same', activation='relu') (net)
    net = MaxPooling2D((3, 3), strides=2)(net)

    net = Conv2D(192, (3, 3), padding='same', activation='relu') (net)
    net = Conv2D(192, (1, 1), padding='same', activation='relu') (net)
    net = Conv2D(10, (1, 1), padding='same', activation='relu') (net)

    net = GlobalAveragePooling2D() (net)
    out = Dense(num_classes, activation='softmax') (net)

    return Model(input, out)

