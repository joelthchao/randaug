import argparse
from itertools import product
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.datasets import cifar10
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from model import build_simple_model
from wide_resnet import build_wide_resnet_model


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


def run_simple_model(log_dir, batch_size=32, optimizer=Adam, lr=0.0001, epochs=20):
    num_classes = 10

    log_name = 'bs_{}_op_{}_lr_{}_ep_{}'.format(batch_size, optimizer.__name__, lr, epochs)
    log_path = os.path.join(log_dir, log_name)

    x_train, y_train, x_test, y_test = get_cifar10_data(num_classes)

    model = build_simple_model(x_train.shape[1:], num_classes)
    model.compile(optimizer=optimizer(lr), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [TensorBoard(log_path)]
    model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=epochs,
              validation_data=(x_test, y_test), callbacks=callbacks)


def run_sgd(log_dir, batch_size=128, lr=0.1, epochs=200, data_augmentation=False):
    num_classes = 10

    log_name = 'wide_resnet_bs_{}_op_{}_lr_manual_ep_{}_ag_{}'.format(
            batch_size, SGD.__name__, epochs, data_augmentation)
    log_path = os.path.join(log_dir, log_name)

    x_train, y_train, x_test, y_test = get_cifar10_data(num_classes)

    model = build_wide_resnet_model(x_train.shape[1:], num_classes)
    optimizer = SGD(lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    class ManualSGDLRCallback(Callback):
        def __init__(self, lr_schedule):
            self.lr_schedule = lr_schedule
            super(Callback, self).__init__()

        def on_epoch_begin(self, epoch, logs=None):
            if epoch in self.lr_schedule:
                self.model.optimizer.lr = self.lr_schedule[epoch]

    callbacks = [TensorBoard(log_path), ManualSGDLRCallback({0:0.1, 80: 0.01, 120: 0.001})]

    if data_augmentation:
        img_gen = ImageDataGenerator(
                width_shift_range=0.125,
                height_shift_range=0.125,
                fill_mode='reflect',
                horizontal_flip=True
        )
        train_data_gen = img_gen.flow(x_train, y_train, batch_size=batch_size)
        steps_per_epoch = (x_train.shape[0] - 1) // batch_size + 1
        model.fit_generator(train_data_gen, steps_per_epoch=steps_per_epoch,
                            verbose=1, epochs=epochs, validation_data=(x_test, y_test), callbacks=callbacks)
    else:
        model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=epochs,
                  validation_data=(x_test, y_test), callbacks=callbacks)


def run(log_dir, batch_size=128, optimizer=Adam, lr=0.001, epochs=20, data_augmentation=False):
    num_classes = 10

    log_name = 'wide_resnet_bs_{}_op_{}_lr_{}_ep_{}_ag_{}'.format(
            batch_size, optimizer.__name__, lr, epochs, data_augmentation)
    log_path = os.path.join(log_dir, log_name)

    x_train, y_train, x_test, y_test = get_cifar10_data(num_classes)

    model = build_wide_resnet_model(x_train.shape[1:], num_classes)
    model.compile(optimizer=optimizer(lr), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [TensorBoard(log_path)]

    if data_augmentation:
        img_gen = ImageDataGenerator(
                width_shift_range=0.125,
                height_shift_range=0.125,
                fill_mode='reflect',
                horizontal_flip=True
        )
        train_data_gen = img_gen.flow(x_train, y_train, batch_size=batch_size)
        steps_per_epoch = (x_train.shape[0] - 1) // batch_size + 1
        model.fit_generator(train_data_gen, steps_per_epoch=steps_per_epoch,
                            verbose=1, epochs=epochs, validation_data=(x_test, y_test), callbacks=callbacks)
    else:
        model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=epochs,
                  validation_data=(x_test, y_test), callbacks=callbacks)


def main():
    parser = argparse.ArgumentParser(description='DDPG pipeline main function')
    parser.add_argument('--gpu', type=int, help='which gpu to train rnn and ddpg', default=0)
    parser.add_argument('--log_dir', type=str, help='where to dump log', default='/media/sdg/joel/rand_aug_exp_log')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    run_sgd(args.log_dir, epochs=200, data_augmentation=True)


if __name__ == '__main__':
    main()
