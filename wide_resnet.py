
from keras.layers import Input
from keras.layers import Conv2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.merge import add


def bottleneck(incoming, count, nb_in_filters, nb_out_filters, dropout=None, subsample=(2, 2)):
    outgoing = wide_basic(incoming, nb_in_filters, nb_out_filters, dropout, subsample)
    for i in range(1, count):
        outgoing = wide_basic(outgoing, nb_out_filters, nb_out_filters, dropout, strides=(1, 1))

    return outgoing


def wide_basic(incoming, nb_in_filters, nb_out_filters, dropout=None, strides=(2, 2)):
    nb_bottleneck_filter = nb_out_filters

    if nb_in_filters == nb_out_filters:
        # conv3x3
        y = BatchNormalization(axis=3)(incoming)
        y = Activation('relu')(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Conv2D(nb_bottleneck_filter, (3, 3), strides=strides, kernel_initializer='he_normal')(y)

        # conv3x3
        y = BatchNormalization(axis=3)(y)
        y = Activation('relu')(y)
        if dropout is not None:
            y = Dropout(dropout)(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Conv2D(nb_bottleneck_filter, (3, 3), strides=(1, 1), kernel_initializer='he_normal')(y)

        return add([incoming, y])

    else:  # Residual Units for increasing dimensions
        # common BN, ReLU
        shortcut = BatchNormalization(axis=3)(incoming)
        shortcut = Activation('relu')(shortcut)

        # conv3x3
        y = ZeroPadding2D((1, 1))(shortcut)
        y = Conv2D(nb_bottleneck_filter, (3, 3), strides=strides, kernel_initializer='he_normal')(y)

        # conv3x3
        y = BatchNormalization(axis=3)(y)
        y = Activation('relu')(y)
        if dropout is not None:
            y = Dropout(dropout)(y)
        y = ZeroPadding2D((1, 1))(y)
        y = Conv2D(nb_out_filters, (3, 3), strides=(1, 1), kernel_initializer='he_normal')(y)

        # shortcut
        shortcut = Conv2D(nb_out_filters, (1, 1), strides=strides, kernel_initializer='he_normal')(shortcut)

        return add([shortcut, y])


def build_wide_resnet_model(input_shape, num_classes):
    n = 4  # depth = 6*n + 4
    k = 4  # widen factor

    img_input = Input(shape=input_shape)

    # one conv at the beginning (spatial size: 32x32)
    x = ZeroPadding2D((1, 1))(img_input)
    x = Conv2D(16, (3, 3))(x)

    # Stage 1 (spatial size: 32x32)
    x = bottleneck(x, n, 16, 16 * k, dropout=0.3, subsample=(1, 1))
    # Stage 2 (spatial size: 16x16)
    x = bottleneck(x, n, 16 * k, 32 * k, dropout=0.3, subsample=(2, 2))
    # Stage 3 (spatial size: 8x8)
    x = bottleneck(x, n, 32 * k, 64 * k, dropout=0.3, subsample=(2, 2))

    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8), strides=(1, 1))(x)
    x = Flatten()(x)
    preds = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=preds)
    return model
