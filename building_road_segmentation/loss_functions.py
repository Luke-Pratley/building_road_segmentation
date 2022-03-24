#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras
import numpy as np


def weighted_dice_loss(weights=1, mask_value=-1):
    """
    Constructs the dice_loss function with weights and masking.

    Input:
        weights: weights that can be passed to balance the categories
        mask_value: The label value that suggests a label should be masked.

    Output:
        dice_loss: the dice loss function to use during training.
    """
    assert np.all(np.array(weights) <= 1)
    assert np.all(np.array(weights) >= 0)

    def dice_loss(y_true, y_pred):
        """
        Calculates the dice loss given predicted and true labels.

        Input:
            y_true: True labels that might be masked.
            y_pred: Predicted labels.
        Output:
            dice_loss_value: The calculated dice dice loss over an image.
        """
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred):
            y_pred = tf.constant(y_pred)
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


def weighted_categorical_crossentropy(weights=1):
    """
    Constructs the categorical cross-entropy function with weights.

    Input:
        weights: weights that can be passed to balance the categories.

    Output:
        categorical_crossentropy: the categorical cross-entropy loss 
                                    function to use during training.
    """

    def categorical_crossentropy(y_true, y_pred):
        """
        Calculates the categorical cross-entropy given predicted and true labels.

        Input:
            y_true: True labels that are not masked.
            y_pred: Predicted labels.
        Output:
            crossentropy_value: The calculated cross-entropy loss per pixel.
        """
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred):
            y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.keras.backend.categorical_crossentropy(
            y_true * tfweights, y_pred)

    return categorical_crossentropy


def weighted_binary_crossentropy(weights=1, mask_value=-1):
    """
    Constructs the binary cross-entropy function with weights and masking.

    Input:
        weights: weights that can be passed to balance the categories
        mask_value: The label value that suggests a label should be masked.

    Output:
        binary_crossentropy: the binary cross-entropy function to use during training.
    """
    assert np.all(np.array(weights) <= 1)
    assert np.all(np.array(weights) >= 0)

    def binary_crossentropy(y_true, y_pred):
        """
        Calculates the binary cross-entropy loss given predicted and true labels.

        Input:
            y_true: True labels that might be masked.
            y_pred: Predicted labels.
        Output:
            binary_crossentropy_value: The calculated binary 
                                        cross-entropy loss over an image.
        """
        tfweights = tf.constant(weights, dtype=y_pred.dtype)
        if not tf.is_tensor(y_pred): 
            y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(
            tf.keras.backend.not_equal(tf.reduce_mean(y_true, axis=(-3, -2)),
                                       mask_value), y_pred.dtype)
        return tf.reduce_sum(
            tf.keras.backend.binary_crossentropy(y_true * tfweights, y_pred) *
            mask[:, tf.newaxis, tf.newaxis, :],
            axis=-1) / tf.reduce_sum(mask, axis=-1)[:, tf.newaxis, tf.newaxis]

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


def iou_metric(y_true, y_pred):
    if not tf.is_tensor(y_pred):
        y_pred = tf.constant(y_pred)
    if not tf.is_tensor(y_true):
        y_pred = tf.constant(y_true)
    y_pred = tf.cast(tf.keras.backend.greater(y_pred, 0.5), dtype=y_true.dtype)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(-3, -2))
    union = tf.reduce_sum(tf.cast(tf.keras.backend.greater(y_pred + y_true, 0),
                                  dtype=y_true.dtype),
                          axis=(-3, -2))
    mask = tf.keras.backend.not_equal(union, 0)
    union = tf.boolean_mask(union, mask)
    intersection = tf.boolean_mask(intersection, mask)
    return intersection / union


def masked_accuracy(mask_value=-1):

    def accuracy(y_true, y_pred):
        if not tf.is_tensor(y_pred):
            y_pred = tf.constant(y_pred)
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
