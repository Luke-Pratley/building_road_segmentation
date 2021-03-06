#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import building_road_segmentation.loss_functions as loss_functions


def test_constant_weighted_categorical_cross_entropy():
    """
    Test the output for the categorical cross entropy.
    """
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
    """
    Test the output for the weighted dice loss function.
    """
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = true_input / np.sum(true_input, axis=-1)[:, :, :, np.newaxis]
    test_input = test_input / np.sum(test_input, axis=-1)[:, :, :, np.newaxis]
    weights = np.ones(shape=test_input.shape[-1])

    output = 1 - 2 * (
        np.sum(test_input * true_input * weights, axis=(-3, -2, -1)) +
        1e-13) / (np.sum(
            (test_input + true_input) * weights, axis=(-3, -2, -1)) + 1e-13)

    loss_fn = loss_functions.weighted_dice_loss(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)


def test_random_weighted_categorical_cross_entropy():
    """
    Test the output for the weighted cross entropy
    with random weights applied.
    """
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
    """
    Test the output for the weighted dice loss function
    with random weights applied.
    """
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = true_input / np.sum(true_input, axis=-1)[:, :, :, np.newaxis]
    test_input = test_input / np.sum(test_input, axis=-1)[:, :, :, np.newaxis]

    weights = np.random.uniform(0, 1, size=test_input.shape)

    output = 1 - 2 * (
        np.sum(test_input * true_input * weights, axis=(-3, -2, -1)) +
        1e-13) / (np.sum(
            (test_input + true_input) * weights, axis=(-3, -2, -1)) + 1e-13)

    loss_fn = loss_functions.weighted_dice_loss(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)


def test_random_weighted_masked_dice_loss():
    """
    Test the output for the weighted dice loss function
    with random weights applied and censoring of labels that are missing.
    """
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input[:, :, :, 0] = -1
    weights = np.random.uniform(0, 1, size=test_input.shape)

    output = 1 - 2 * (np.sum(test_input[:, :, :, 1:] *
                             true_input[:, :, :, 1:] * weights[:, :, :, 1:],
                             axis=(-3, -2, -1)) +
                      1e-13) / (np.sum(
                          (test_input[:, :, :, 1:] + true_input[:, :, :, 1:]) *
                          weights[:, :, :, 1:],
                          axis=(-3, -2, -1)) + 1e-13)

    loss_fn = loss_functions.weighted_dice_loss(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-6, atol=1e-6)


def test_random_weighted_binary_cross_entropy():
    """
    Test the output for the weighted binary cross entropy
    with random weights.
    """
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    weights = np.random.uniform(0, 1, size=test_input.shape) / 2
    norm = weights * true_input + (1 - true_input * weights)
    weights = weights / norm
    output = -np.mean(np.log(test_input) * true_input * weights +
                      np.log(1 - test_input) * (1 - true_input * weights),
                      axis=(-1))

    loss_fn = loss_functions.weighted_binary_crossentropy(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    # there is an unknown epsilon in the tf calculation,
    # so we don't expect them to be exactly close
    assert np.allclose(output, test_output, rtol=1e-3, atol=1e-3)


def test_random_weighted_masked_binary_cross_entropy():
    """
    Test the output for the weighted binary cross entropy
    with random weights applied and censoring of labels that are missing.
    """
    np.random.seed(0)
    class_num = 5
    test_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    true_input = np.random.uniform(0, 1, size=(1, 128, 128, class_num))
    weights = np.random.uniform(0, 1, size=test_input.shape) / 2
    true_input[:, :, :, 0] = -1
    norm = weights * true_input + (1 - true_input * weights)
    weights = weights / norm
    output = -np.mean(
        np.log(test_input[:, :, :, 1:]) * true_input[:, :, :, 1:] *
        weights[:, :, :, 1:] + np.log(1 - test_input[:, :, :, 1:]) *
        (1 - true_input[:, :, :, 1:] * weights[:, :, :, 1:]),
        axis=(-1))

    loss_fn = loss_functions.weighted_binary_crossentropy(weights)

    test_output = loss_fn(true_input, test_input).numpy()
    assert test_output.shape == output.shape
    assert np.allclose(output, test_output, rtol=1e-3, atol=1e-3)
