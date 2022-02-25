#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import tensorflow as tf
import tensorflow.keras


import building_road_segmentation.unet_factory as unet_factory
import building_road_segmentation.optimization_factory as optimization_factory
import building_road_segmentation.data_generator as data_generator
import numpy as np


def test_conv_block():

    block = unet_factory.ConvBlock(number_of_start_kernels=8,
                                   kernel_shape=(3, 3),
                                   activation=tf.nn.relu)
    assert isinstance(block, tf.keras.Model)
    inp = tf.constant(np.random.normal(0, 1, (4, 16, 16, 3)), dtype=np.float32)
    out = block(inp)
    assert (out.shape == (4, 16, 16, 8))
    assert (out.shape.as_list() == [4, 16, 16, 8])
    blocks = block.layers
    assert len(blocks) == 6
    assert isinstance(blocks[0], tf.keras.layers.Conv2D)
    assert isinstance(blocks[1], tf.keras.layers.BatchNormalization)
    assert isinstance(blocks[2], tf.keras.layers.Activation)
    assert isinstance(blocks[3], tf.keras.layers.Conv2D)
    assert isinstance(blocks[4], tf.keras.layers.BatchNormalization)
    assert isinstance(blocks[5], tf.keras.layers.Activation)


def test_downlayer():

    x = unet_factory.DownLayer(number_of_start_kernels=8,
                               kernel_shape=(3, 3),
                               activation=tf.nn.relu,
                               pooling_amount=2,
                               dropout_rate=0.5)
    assert isinstance(x, tf.keras.Model)
    inp = tf.constant(np.random.normal(0, 1, (4, 16, 16, 3)), dtype=np.float32)
    out = x(inp)
    assert (out.shape.as_list() == [4, 8, 8, 8])
    blocks = x.layers
    assert len(blocks) == 3
    assert isinstance(blocks[0], tf.keras.layers.MaxPooling2D)
    assert isinstance(blocks[1], tf.keras.layers.Dropout)
    assert isinstance(blocks[2], unet_factory.ConvBlock)


def test_uplayer():

    x = unet_factory.UpLayer(number_of_start_kernels=8,
                             kernel_shape=(3, 3),
                             activation=tf.nn.relu,
                             pooling_amount=2,
                             dropout_rate=0.5)
    assert isinstance(x, tf.keras.Model)
    inp1 = tf.constant(np.random.normal(0, 1, (4, 8, 8, 3)), dtype=np.float32)
    inp2 = tf.constant(np.random.normal(0, 1, (4, 16, 16, 3)),
                       dtype=np.float32)
    out = x(inp1, inp2)
    assert (out.shape.as_list() == [4, 16, 16, 8])
    blocks = x.layers
    assert len(blocks) == 4
    assert isinstance(blocks[0], tf.keras.layers.UpSampling2D)
    assert isinstance(blocks[1], tf.keras.layers.Concatenate)
    assert isinstance(blocks[2], tf.keras.layers.Dropout)
    assert isinstance(blocks[3], unet_factory.ConvBlock)


def test_basic_unet():
    unet_levels = 6
    number_of_categories = 1
    x = unet_factory.BasicUnet(number_of_categories=number_of_categories,
                               unet_levels=unet_levels,
                               number_of_start_kernels=4,
                               kernel_shape=(3, 3),
                               activation='relu',
                               pooling_amount=2,
                               dropout_rate=0.5)
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


def test_trainer():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(16, ), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    data_gen = data_generator.test_data(np.random.normal(0, 16, (128, 16)),
                                        np.random.uniform(0, 1, (128, 1)), 4)

    trainer = optimization_factory.Trainer(model, loss_fn, optimizer,
                                           train_acc_metric, val_acc_metric,
                                           )

    trainer.fit(data_gen, None, 2)
