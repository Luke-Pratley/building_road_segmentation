#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras
import numpy as np


def weighted_dice_loss(weights):

    def dice_loss(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return 1 - 2 * (tf.reduce_sum(
            tf.math.multiply(tf.math.multiply(tfweights, y_true), y_pred),
            axis=(-3, -2, -1)) + 1e-13) / (tf.reduce_sum(
                tf.multiply(y_true + y_pred, tfweights), axis=(-3, -2, -1)) + 1e-13)

    return dice_loss



def weighted_categorical_crossentropy(weights):

    def categorical_crossentropy(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.keras.backend.categorical_crossentropy(
            y_true * tfweights, y_pred)

    return categorical_crossentropy


def weighted_binary_crossentropy(weights):
    assert np.all(weights <= 1)
    assert np.all(weights >= 0)
    def binary_crossentropy(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        tfnot_weights = 1 - tfweights
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        not_y_true = 1 - y_true
        not_y_pred = 1 - y_pred
        return -tf.reduce_mean(
            (y_true * tfweights) * tf.math.log(y_pred) +
            (not_y_true * tfnot_weights) * tf.math.log(not_y_pred),
            axis=(-1))

    return binary_crossentropy


def intersection_over_union(y_true, y_pred):
    intersection = float(np.sum(y_true * y_pred))
    union = float(np.sum((y_true == 1) | (y_pred == 1)))
    return intersection / union
