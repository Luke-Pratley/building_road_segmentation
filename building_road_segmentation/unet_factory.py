#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

def conv_block(inp):
    conv = tf.keras.layers.Conv2D(number_of_start_kernels, kernel_shape, padding='same')(inp)
    conv = tf.keras.BatchNorm()(conv)
    conv = tf.keras.layers.Activation(activation)(conv)
    conv = tf.keras.layers.Conv2D(number_of_start_kernels, kernel_shape, padding='same')(conv)
    conv = tf.keras.BatchNorm()(conv)
    conv = tf.keras.layers.Activation(activation)(conv)
    return conv


def downlayer(inp):
    pool = tf.keras.layers.MaxPooling((pooling_amount, pooling_amount))(inp)
    pool = tf.keras.layers.Dropout(dropout_rate)(pool)
    return conv_block(pool)

def uplayer(inp1, inp2):
    transconv = tf.keras.layers.Conv2DTranspose(number_of_start_kernels * pooling_amount**k, (3, 3), activation='relu', strides=(2, 2), padding='same')(inp1)
    uconv = tf.concatenate([inp1, inp2])
    uconv = tf.keras.layers.Dropout(dropout_rate)(uconv)
    return conv_block(uconv)

def create_basic_unet(input_shape, output_shape, number_of_start_kernels, kernel_shape, conv_activation, pooling_amount, dropout_rate, u_depth):
    conv_layers = []
    input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
    x = input_layer(x)

    for k in range(u_depth):
        x = downlayer(x)
        conv_layer.append(x)

    x = conv_layer[-1]
    for k in reversed(range(u_depth)):
        x = uplayer(x, conv_layer[k - 1])

    x = tf.keras.layers.Conv2D(num_of_classes, 1, acivation='softmax',padding='same')(x)
    return keras.model.Model(input_layer, x)
