#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras
import numpy as np


def weighted_dice_loss(weights, mask_value=-1):

    def dice_loss(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(
            tf.keras.backend.not_equal(tf.reduce_mean(y_true, axis=(-3, -2)),
                                       mask_value), y_pred.dtype)
        return 1 - 2 * (tf.reduce_sum(
            tfweights * y_true * y_pred * mask[:, tf.newaxis, tf.newaxis, :],
            axis=(-3, -2, -1)) + 1e-13) / (tf.reduce_sum(
                (y_true + y_pred) * tfweights *
                mask[:, tf.newaxis, tf.newaxis, :],
                axis=(-3, -2, -1)) + 1e-13)

    return dice_loss


def weighted_categorical_crossentropy(weights):

    def categorical_crossentropy(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.keras.backend.categorical_crossentropy(
            y_true * tfweights, y_pred)

    return categorical_crossentropy


def weighted_binary_crossentropy(weights, mask_value=-1):
    assert np.all(weights <= 1)
    assert np.all(weights >= 0)

    def binary_crossentropy(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        tfnot_weights = 1 - tfweights
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        not_y_true = 1 - y_true
        not_y_pred = 1 - y_pred
        mask = tf.cast(
            tf.keras.backend.not_equal(tf.reduce_mean(y_true, axis=(-3, -2)),
                                       mask_value), y_pred.dtype)
        return tf.reduce_sum(tf.keras.backend.binary_crossentropy(
            y_true * tfweights, y_pred) * mask[:, tf.newaxis, tf.newaxis, :],
                             axis=-1) / tf.reduce_sum(
                                 mask, axis=-1)[:, tf.newaxis, tf.newaxis]

    return binary_crossentropy


def intersection_over_union(y_true, y_pred, masked_value=-1):
    mask = (y_true != masked_value)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    if len(y_true) > 0:
        intersection = float(np.sum(y_true * y_pred))
        union = float(np.sum((y_true == 1) | (y_pred == 1)))
        return intersection / union
    return np.nan


def masked_accuracy(mask_value=-1):

    def accuracy(y_true, y_pred):
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(tf.keras.backend.not_equal(y_true, mask_value),
                       y_pred.dtype)
        y_pred = tf.math.round(y_pred)
        y_pred = tf.boolean_mask(y_pred, mask)
        y_true = tf.boolean_mask(y_true, mask)
        return tf.reduce_mean(tf.cast(tf.keras.backend.equal(y_true, y_pred),
                                      dtype=y_true.dtype),
                              axis=-1)

    return accuracy


def masked_class_accuracy(mask_value=-1, class_num=0):

    def accuracy(y_true, y_pred):
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(tf.keras.backend.not_equal(y_true, mask_value),
                       y_pred.dtype)
        y_pred = tf.math.round(y_pred)
        y_pred = tf.boolean_mask(y_pred, mask)
        y_true = tf.boolean_mask(y_true, mask)
        return tf.cast(tf.keras.backend.equal(y_true, y_pred),
                       dtype=y_true.dtype)[:, :, :, class_num]

    return accuracy
