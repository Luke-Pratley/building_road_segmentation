#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
import tensorflow as tf
import math
import tensorflow.keras.utils
import numpy as np
import PIL
import tensorflow.image


class READ_DATA(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(
            [np.array(PIL.Image.open(im)) / 255. for im in batch_x],
            dtype=np.float32), np.array([
                np.load(file_name).astype(np.float32) for file_name in batch_y
            ])


class READ_AND_AUGMENT_DATA(tf.keras.utils.Sequence):

    class Augment(Enum):
        ORIGINAL = 1
        LRFLIP = 2
        UDFLIP = 3
        ROT90 = 4
        ROT270 = 5

    def __init__(self, x_set, y_set, batch_size):
        self.batch_size = batch_size
        self.x = []
        self.y = []
        for aug in self.Augment:
            self.x += [(k, aug) for k in x_set]
            self.y += [(k, aug) for k in y_set]

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def apply_augment(self, image, aug):
        if aug is self.Augment.ORIGINAL:
            return image
        if aug is self.Augment.LRFLIP:
            return tf.image.flip_left_right(image)
        if aug is self.Augment.UDFLIP:
            return tf.image.flip_up_down(image)
        if aug is self.Augment.ROT90:
            return tf.image.rot90(image)
        if aug is self.Augment.ROT270:
            return tf.image.rot90(image, k=3)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            self.apply_augment(np.array(PIL.Image.open(im[0])) / 255., im[1])
            for im in batch_x
        ],
                        dtype=np.float32), np.array([
                            self.apply_augment(
                                np.load(file_name[0]).astype(np.float32),
                                file_name[1]) for file_name in batch_y
                        ])


class TEST_DATA(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
