#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import tensorflow as tf
import tensorflow.keras

import numpy as np
import building_road_segmentation.loss_functions as loss_functions


def test_constant_weighted_categorical_cross_entropy():
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = true_input / np.sum(true_input, axis=-1)[:, :, :, np.newaxis]
    test_input = test_input / np.sum(test_input, axis=-1)[:, :, :, np.newaxis]
    weights = np.ones(shape=test_input.shape[-1])
    output = -np.sum(np.log(test_input) * true_input * weights, axis=(-1))

    loss_fn = loss_functions.weighted_categorical_crossentropy(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)


def test_constant_weighted_dice_loss():
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = true_input / np.sum(true_input, axis=-1)[:, :, :, np.newaxis]
    test_input = test_input / np.sum(test_input, axis=-1)[:, :, :, np.newaxis]
    weights = np.ones(shape=test_input.shape[-1])

    output = 1 - 2 * (np.sum(
        test_input * true_input * weights, axis=(-1)) + 1e-13) / (np.sum(
            (test_input + true_input) * weights, axis=(-1)) + 1e-13)

    loss_fn = loss_functions.weighted_dice_loss(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)


def test_random_weighted_categorical_cross_entropy():
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = true_input / np.sum(true_input, axis=-1)[:, :, :, np.newaxis]
    test_input = test_input / np.sum(test_input, axis=-1)[:, :, :, np.newaxis]
    weights = np.random.uniform(0, 1, size=test_input.shape)
    output = -np.sum(np.log(test_input) * true_input * weights, axis=(-1))

    loss_fn = loss_functions.weighted_categorical_crossentropy(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)


def test_random_weighted_dice_loss():
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = true_input / np.sum(true_input, axis=-1)[:, :, :, np.newaxis]
    test_input = test_input / np.sum(test_input, axis=-1)[:, :, :, np.newaxis]

    weights = np.random.uniform(0, 1, size=test_input.shape)

    output = 1 - 2 * (np.sum(
        test_input * true_input * weights, axis=(-1)) + 1e-13) / (np.sum(
            (test_input + true_input) * weights, axis=(-1)) + 1e-13)

    loss_fn = loss_functions.weighted_dice_loss(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)


def test_random_weighted_binary_cross_entropy():
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = true_input / np.sum(true_input, axis=-1)[:, :, :, np.newaxis]
    test_input = test_input / np.sum(test_input, axis=-1)[:, :, :, np.newaxis]
    weights = np.random.uniform(0, 1, size=test_input.shape)
    notweights = np.random.uniform(0, 1, size=test_input.shape)
    output = -np.mean(
        np.log(test_input + 1e-13) * true_input * weights + 
            np.log(1 - test_input + 1e-13) * (1 - true_input) * notweights, axis=(-1))

    loss_fn = loss_functions.weighted_binary_crossentropy(weights, notweights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)


def test_random_weighted_binary_dice_loss():
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = true_input / np.sum(true_input, axis=-1)[:, :, :, np.newaxis]
    test_input = test_input / np.sum(test_input, axis=-1)[:, :, :, np.newaxis]

    weights = np.random.uniform(0, 1, size=test_input.shape)
    notweights = np.random.uniform(0, 1, size=test_input.shape)

    output = 1 - 2 * (np.sum(test_input * true_input * weights +
                             (1 - test_input) * (1 - true_input) * notweights,
                             axis=(-1)) + 1e-13) / (np.sum(
                                 (test_input + true_input) * weights +
                                 (2 - test_input - true_input) * notweights,
                                 axis=(-1)) + 1e-13)

    loss_fn = loss_functions.weighted_binary_dice_loss(weights, notweights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)
