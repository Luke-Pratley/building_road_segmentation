#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.applications.efficientnet

import building_road_segmentation.unet_factory as unet_factory
import building_road_segmentation.optimization_factory as optimization_factory
import building_road_segmentation.data_generator as data_generator
import numpy as np


def test_basic_unet():
    """
    We test properties of a constructed UNet.
    """
    unet_levels = 6
    number_of_categories = 1
    # construct a UNet
    x = unet_factory.BasicUNet(
        number_of_categories=number_of_categories,
        unet_levels=unet_levels,
        number_of_start_kernels=4,
        kernel_shape=(3, 3),
        activation='relu',
        final_activation='sigmoid',
        pooling_amount=2,
        dropout_rate=0.5,
        residual=True,
        kernel_initializer=tf.keras.initializers.he_normal())
    inp = tf.constant(np.random.normal(0, 1, (4, 128, 128, 3)),
                      dtype=np.float32)
    output = x(inp)
    print(x.summary())
    # test output size
    assert (output.shape.as_list() == [4, 128, 128, 1])
    blocks = x.layers
    print([type(b) for b in blocks])
    # test the layers it holds.
    assert len(blocks) == 2 * unet_levels + 2
    assert isinstance(blocks[-1], tf.keras.layers.Conv2D)
    assert isinstance(blocks[-2], tf.keras.layers.Conv2D)
    for b in range(unet_levels):
        assert isinstance(blocks[b], unet_factory.DownLayer)
    for b in range(unet_levels):
        assert isinstance(blocks[b + unet_levels], unet_factory.UpLayer)
    # test training it.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    x.compile(optimizer=optimizer, loss=loss_fn)
    data_gen = data_generator.TEST_DATA(
        np.random.uniform(0, 1, (16, 128, 128, 3)),
        np.random.uniform(0, 1, (16, 128, 128, 1)), 4)
    x.fit(data_gen, epochs=2)


def test_efficientnet_unet():
    """
    We test properties of an EfficientNet with UNet.
    """
    unet_levels = 5
    number_of_categories = 1
    model_name = tensorflow.keras.applications.efficientnet.EfficientNetB4
    # construct the unet
    x = unet_factory.EfficientNetUNet(
        efficientnet=model_name,
        number_of_categories=number_of_categories,
        unet_levels=unet_levels,
        number_of_start_kernels=4,
        kernel_shape=(3, 3),
        activation='relu',
        final_activation='sigmoid',
        pooling_amount=2,
        dropout_rate=0.5,
        residual=True,
        kernel_initializer=tf.keras.initializers.he_normal())
    inp = tf.constant(np.random.normal(0, 1, (4, 128, 128, 3)),
                      dtype=np.float32)
    output = x(inp)
    print(x.summary())
    # test the output
    assert (output.shape.as_list() == [4, 128, 128, 1])
    blocks = x.layers
    print([type(b) for b in blocks])
    # test the layers
    assert len(blocks) == unet_levels + 1
    assert isinstance(blocks[-1], tf.keras.layers.Conv2D)
    assert isinstance(blocks[-2], tf.keras.Model)
    for b in range(unet_levels - 1):
        assert isinstance(blocks[b], unet_factory.UpLayer)
    # test running the fitting process
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    x.compile(optimizer=optimizer, loss=loss_fn)
    data_gen = data_generator.TEST_DATA(
        np.random.uniform(0, 1, (16, 128, 128, 3)),
        np.random.uniform(0, 1, (16, 128, 128, 1)), 4)
    x.fit(data_gen, epochs=2)


def test_attention_unet():
    """
    We test properties of the Attention UNet.
    """
    unet_levels = 6
    number_of_categories = 1
    # construct the unet
    x = unet_factory.AttentionUNet(
        number_of_categories=number_of_categories,
        unet_levels=unet_levels,
        number_of_start_kernels=4,
        kernel_shape=(3, 3),
        activation='relu',
        final_activation='sigmoid',
        pooling_amount=2,
        dropout_rate=0.5,
        residual=True,
        kernel_initializer=tf.keras.initializers.he_normal())
    inp = tf.constant(np.random.normal(0, 1, (4, 128, 128, 3)),
                      dtype=np.float32)
    output = x(inp)
    print(x.summary())
    # test the output size
    assert (output.shape.as_list() == [4, 128, 128, 1])
    blocks = x.layers
    print([type(b) for b in blocks])
    # test the layers
    assert len(blocks) == 3 * unet_levels + 2
    for b in range(unet_levels):
        assert isinstance(blocks[b], unet_factory.DownLayer)
    for b in range(unet_levels):
        assert isinstance(blocks[b + unet_levels], unet_factory.UpLayer)
    assert isinstance(blocks[unet_levels * 2], tf.keras.layers.Conv2D)
    assert isinstance(blocks[unet_levels * 2 + 1], tf.keras.layers.Conv2D)
    for b in range(unet_levels):
        assert isinstance(blocks[b + 2 * unet_levels + 2],
                          unet_factory.AttentionGate)
    # test running the optimization process
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    x.compile(optimizer=optimizer, loss=loss_fn)
    data_gen = data_generator.TEST_DATA(
        np.random.uniform(0, 1, (16, 128, 128, 3)),
        np.random.uniform(0, 1, (16, 128, 128, 1)), 4)
    x.fit(data_gen, epochs=2)


def test_trainer():
    """
    Test the custom object called Trainer runs.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(16, ), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()

    data_gen = data_generator.TEST_DATA(np.random.normal(0, 16, (128, 16)),
                                        np.random.uniform(0, 1, (128, 1)), 4)

    trainer = optimization_factory.Trainer(
        model,
        loss_fn,
        optimizer,
        {"acc": train_acc_metric},
    )

    trainer.fit(data_gen, None, 2)
