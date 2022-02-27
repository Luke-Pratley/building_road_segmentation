#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import tensorflow as tf
import tensorflow.keras

import building_road_segmentation.unet_factory as unet_factory
import numpy as np


def test_conv_block():

    block = unet_factory.ConvBlock(
        number_of_start_kernels=8,
        kernel_shape=(3, 3),
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.he_normal())
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


def test_attention_gate():

    block = unet_factory.AttentionGate(
        num_filters=8, pooling_amount=2, kernel_initializer=tf.keras.initializers.Constant(1))
    assert isinstance(block, tf.keras.Model)
    inp1 = tf.constant(np.random.normal(0, 1, (4, 16, 16, 8)), dtype=np.float32)
    inp2 = tf.constant(np.random.normal(0, 1, (4, 8, 8, 16)), dtype=np.float32)
    out = block([inp1, inp2])
    assert (out.shape == (4, 16, 16, 8))
    assert (out.shape.as_list() == [4, 16, 16, 8])
    blocks = block.layers
    assert len(blocks) == 8
    assert isinstance(blocks[0], tf.keras.layers.Conv2D)
    assert isinstance(blocks[1], tf.keras.layers.Conv2D)
    assert isinstance(blocks[2], tf.keras.layers.Add)
    assert isinstance(blocks[3], tf.keras.layers.Activation)
    assert isinstance(blocks[4], tf.keras.layers.Conv2D)
    assert isinstance(blocks[5], tf.keras.layers.Activation)
    assert isinstance(blocks[6], tf.keras.layers.UpSampling2D)
    assert isinstance(blocks[7], tf.keras.layers.Multiply)


def test_downlayer():

    x = unet_factory.DownLayer(
        number_of_start_kernels=8,
        kernel_shape=(3, 3),
        activation=tf.nn.relu,
        pooling_amount=2,
        dropout_rate=0.5,
        kernel_initializer=tf.keras.initializers.he_normal())
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

    x = unet_factory.UpLayer(
        number_of_start_kernels=8,
        kernel_shape=(3, 3),
        activation=tf.nn.relu,
        pooling_amount=2,
        dropout_rate=0.5,
        kernel_initializer=tf.keras.initializers.he_normal())
    assert isinstance(x, tf.keras.Model)
    inp1 = tf.constant(np.random.normal(0, 1, (4, 8, 8, 3)), dtype=np.float32)
    inp2 = tf.constant(np.random.normal(0, 1, (4, 16, 16, 3)),
                       dtype=np.float32)
    out = x([inp1, inp2])
    assert (out.shape.as_list() == [4, 16, 16, 8])
    blocks = x.layers
    assert len(blocks) == 4
    assert isinstance(blocks[0], tf.keras.layers.UpSampling2D)
    assert isinstance(blocks[1], tf.keras.layers.Concatenate)
    assert isinstance(blocks[2], tf.keras.layers.Dropout)
    assert isinstance(blocks[3], unet_factory.ConvBlock)


