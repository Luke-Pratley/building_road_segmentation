#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import tensorflow as tf

import building_road_segmentation
import numpy as np



def test_conv_block():

    block = building_road_segmentation.ConvBlock(number_of_start_kernels=8,
                                                 kernel_shape=(3, 3),
                                                 activation=tf.nn.relu)
    assert isinstance(block, tf.keras.Model)
    inp = tf.constant(np.random.normal(0, 1, (4, 16, 16, 3)), dtype=np.float32)
    out = block(inp)
    assert (out.shape == (4, 16, 16, 8))
    assert (out.shape.as_list() == [4, 16, 16, 8])
    blocks = [block.get_layer(index=b) for b in range(6)]
    assert isinstance(blocks[0], tf.keras.layers.Conv2D)
    assert isinstance(blocks[1], tf.keras.layers.BatchNormalization)
    assert isinstance(blocks[2], tf.keras.layers.Activation)
    assert isinstance(blocks[3], tf.keras.layers.Conv2D)
    assert isinstance(blocks[4], tf.keras.layers.BatchNormalization)
    assert isinstance(blocks[5], tf.keras.layers.Activation)

def test_downlayer():

    x = building_road_segmentation.DownLayer(number_of_start_kernels=8,
                                             kernel_shape=(3, 3),
                                             activation=tf.nn.relu,
                                             pooling_amount=2,
                                             dropout_rate=0.5)
    assert isinstance(x, tf.keras.Model)
    inp = tf.constant(np.random.normal(0, 1, (4, 16, 16, 3)), dtype=np.float32)
    out = x(inp)
    assert (out.shape.as_list() == [4, 8, 8, 8])
    blocks = [x.get_layer(index=b) for b in range(3)]
    assert isinstance(blocks[0], tf.keras.layers.MaxPooling2D)
    assert isinstance(blocks[1], tf.keras.layers.Dropout)
    assert isinstance(blocks[2], building_road_segmentation.ConvBlock)

def test_uplayer():

    x = building_road_segmentation.UpLayer(number_of_start_kernels=8,
                                             kernel_shape=(3, 3),
                                             activation=tf.nn.relu,
                                             pooling_amount=2,
                                             dropout_rate=0.5)
    assert isinstance(x, tf.keras.Model)
    inp1 = tf.constant(np.random.normal(0, 1, (4, 8, 8, 3)), dtype=np.float32)
    inp2 = tf.constant(np.random.normal(0, 1, (4, 16, 16, 3)), dtype=np.float32)
    out = x(inp1, inp2)
    assert (out.shape.as_list() == [4, 16, 16, 8])
    blocks = [x.get_layer(index=b) for b in range(4)]
    assert isinstance(blocks[0], tf.keras.layers.UpSampling2D)
    assert isinstance(blocks[1], tf.keras.layers.Concatenate)
    assert isinstance(blocks[2], tf.keras.layers.Dropout)
    assert isinstance(blocks[3], building_road_segmentation.ConvBlock)
