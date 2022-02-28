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
            tf.math.multiply(tf.math.multiply(
                tfweights, y_true), y_pred), axis=-1) + 1e-13) / (tf.reduce_sum(
                    tf.multiply(y_true + y_pred, tfweights), axis=-1) + 1e-13)

    return dice_loss


def weighted_categorical_crossentropy(weights):

    def categorical_crossentropy(y_true, y_pred):
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.keras.backend.categorical_crossentropy(
            y_true * tfweights, y_pred)

    return cateogrical_cross_entropy
