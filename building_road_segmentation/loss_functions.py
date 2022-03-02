#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras


def weighted_dice_loss(weights):

    def dice_loss(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return 1 - 2 * (tf.reduce_sum(
            tf.math.multiply(tf.math.multiply(tfweights, y_true), y_pred),
            axis=-1) + 1e-13) / (tf.reduce_sum(
                tf.multiply(y_true + y_pred, tfweights), axis=-1) + 1e-13)

    return dice_loss


def weighted_binary_dice_loss(weights, not_weights):

    def dice_loss(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        tfnot_weights = tf.constant(not_weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        not_y_true = 1 - y_true
        not_y_pred = 1 - y_pred
        return 1 - 2 * (tf.reduce_sum(
            tf.math.multiply(tf.math.multiply(tfweights, y_true), y_pred) +
            tf.math.multiply(tf.math.multiply(tfnot_weights, not_y_true),
                             not_y_pred),
            axis=-1) + 1e-13) / (tf.reduce_sum(
                tf.multiply(y_true + y_pred, tfweights) +
                tf.multiply(not_y_true + not_y_pred, tfnot_weights),
                axis=-1) + 1e-13)

    return dice_loss


def weighted_categorical_crossentropy(weights):

    def categorical_crossentropy(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.keras.backend.categorical_crossentropy(
            y_true * tfweights, y_pred)

    return categorical_crossentropy
