import argparse
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from model import build_simple_model


def get_cifar10_data(num_classes, sub_pixel_mean=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if sub_pixel_mean:
        mean_pixel = np.mean(x_train, axis=0)
        x_train = x_train - mean_pixel
        x_test = x_test - mean_pixel

    return x_train, y_train, x_test, y_test


def main():
    parser = argparse.ArgumentParser(description='DDPG pipeline main function')
    parser.add_argument('--gpu', type=int, help='which gpu to train rnn and ddpg', default=0)
    parser.add_argument('--log_dir', type=str, help='where to dump log', default='/media/sdg/joel/rand_aug_exp_log')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    batch_size = 128
    num_classes = 10
    epochs = 20
    lr = 0.001

    log_name = 'bs_{}_lr_{}'.format(batch_size, lr)
    log_path = os.path.join(args.log_dir, log_name)

    x_train, y_train, x_test, y_test = get_cifar10_data(num_classes)

    model = build_simple_model(x_train.shape[1:], num_classes)
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [TensorBoard(log_path)]
    model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs,
              validation_data=(x_test, y_test), callbacks=callbacks)


if __name__ == '__main__':
    main()
