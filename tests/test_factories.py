#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import tensorflow as tf
import tensorflow.keras

import building_road_segmentation.unet_factory as unet_factory
import building_road_segmentation.optimization_factory as optimization_factory
import building_road_segmentation.data_generator as data_generator
import numpy as np


def test_basic_unet():
    unet_levels = 6
    number_of_categories = 1
    x = unet_factory.BasicUNet(
        number_of_categories=number_of_categories,
        unet_levels=unet_levels,
        number_of_start_kernels=4,
        kernel_shape=(3, 3),
        activation='relu',
        pooling_amount=2,
        dropout_rate=0.5,
        kernel_initializer=tf.keras.initializers.he_normal())
    inp = tf.constant(np.random.normal(0, 1, (4, 128, 128, 3)),
                      dtype=np.float32)
    output = x(inp)
    print(x.summary())
    assert (output.shape.as_list() == [4, 128, 128, 1])
    blocks = x.layers
    print([type(b) for b in blocks])
    assert len(blocks) == 2 * unet_levels + 2
    assert isinstance(blocks[-1], tf.keras.layers.Conv2D)
    assert isinstance(blocks[-2], tf.keras.layers.Conv2D)
    for b in range(unet_levels):
        assert isinstance(blocks[b], unet_factory.DownLayer)
    for b in range(unet_levels):
        assert isinstance(blocks[b + unet_levels], unet_factory.UpLayer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    x.compile(optimizer=optimizer, loss=loss_fn)
    data_gen = data_generator.TEST_DATA(
        np.random.uniform(0, 1, (16, 128, 128, 3)),
        np.random.uniform(0, 1, (16, 128, 128, 1)), 4)
    x.fit(data_gen, epochs=2)

def test_attention_unet():
    unet_levels = 6
    number_of_categories = 1
    x = unet_factory.AttentionUNet(
        number_of_categories=number_of_categories,
        attention_intermediate_dim=4,
        unet_levels=unet_levels,
        number_of_start_kernels=4,
        kernel_shape=(3, 3),
        activation='relu',
        pooling_amount=2,
        dropout_rate=0.5,
        kernel_initializer=tf.keras.initializers.he_normal())
    inp = tf.constant(np.random.normal(0, 1, (4, 128, 128, 3)),
                      dtype=np.float32)
    output = x(inp)
    print(x.summary())
    assert (output.shape.as_list() == [4, 128, 128, 1])
    blocks = x.layers
    print([type(b) for b in blocks])
    assert len(blocks) == 2 * unet_levels + 2
    assert isinstance(blocks[-1], tf.keras.layers.Conv2D)
    assert isinstance(blocks[-2], tf.keras.layers.Conv2D)
    for b in range(unet_levels):
        assert isinstance(blocks[b], unet_factory.DownLayer)
    for b in range(unet_levels):
        assert isinstance(blocks[b + unet_levels], unet_factory.UpLayer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    x.compile(optimizer=optimizer, loss=loss_fn)
    data_gen = data_generator.TEST_DATA(
        np.random.uniform(0, 1, (16, 128, 128, 3)),
        np.random.uniform(0, 1, (16, 128, 128, 1)), 4)
    x.fit(data_gen, epochs=2)


def test_trainer():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(16, ), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    data_gen = data_generator.TEST_DATA(np.random.normal(0, 16, (128, 16)),
                                        np.random.uniform(0, 1, (128, 1)), 4)

    trainer = optimization_factory.Trainer(
        model,
        loss_fn,
        optimizer,
        {"acc": train_acc_metric},
    )

    trainer.fit(data_gen, None, 2)
