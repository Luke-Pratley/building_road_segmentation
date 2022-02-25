#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class ConvBlock(tf.keras.Model):

    def __init__(self, number_of_start_kernels, kernel_shape, activation,
                 kernel_initializer):
        super(ConvBlock, self).__init__(name='')
        self.conv1 = tf.keras.layers.Conv2D(
            number_of_start_kernels,
            kernel_shape,
            padding='same',
            kernel_initializer=kernel_initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation(activation)

        self.conv2 = tf.keras.layers.Conv2D(
            number_of_start_kernels,
            kernel_shape,
            padding='same',
            kernel_initializer=kernel_initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.activation2 = tf.keras.layers.Activation(activation)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training)
        x = self.activation1(x)

        x = self.conv2(input_tensor)
        x = self.bn2(x, training)
        x = self.activation2(x)
        return x


class DownLayer(tf.keras.Model):

    def __init__(self, number_of_start_kernels, kernel_shape, activation,
                 pooling_amount, dropout_rate, kernel_initializer):
        super(DownLayer, self).__init__(name='')
        self.pool = tf.keras.layers.MaxPooling2D(
            (pooling_amount, pooling_amount))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.convblock = ConvBlock(number_of_start_kernels,
                                   kernel_shape,
                                   activation,
                                   kernel_initializer=kernel_initializer)

    def call(self, input_tensor, training=False):
        x = self.pool(input_tensor)
        x = self.dropout(x, training)
        return self.convblock(x, training)


class UpLayer(tf.keras.Model):

    def __init__(self, number_of_start_kernels, kernel_shape, activation,
                 pooling_amount, dropout_rate, kernel_initializer):
        super(UpLayer, self).__init__(name='')
        # TODO: create option to switch between upsampling and transpose convolution
        self.upsample = tf.keras.layers.UpSampling2D(size=(pooling_amount,
                                                           pooling_amount))
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.convblock = ConvBlock(number_of_start_kernels,
                                   kernel_shape,
                                   activation,
                                   kernel_initializer=kernel_initializer)

    def call(self, input_tensor1, input_tensor2, training=False):
        x = self.upsample(input_tensor1)
        x = self.concat([x, input_tensor2])
        x = self.dropout(x, training)
        x = self.convblock(x, training)
        return x


class BasicUnet(tf.keras.Model):

    def __init__(self,
                 number_of_categories,
                 unet_levels,
                 number_of_start_kernels,
                 kernel_shape,
                 activation,
                 pooling_amount,
                 dropout_rate,
                 kernel_initializer=tf.keras.initializers.he_normal()):
        super(BasicUnet, self).__init__(name='')
        assert unet_levels > 0, "Unet levels is less than 1"
        assert number_of_categories > 0, "number of classes/categories less than 1"
        self.unet_levels = unet_levels
        self.down_blocks = []
        self.up_blocks = []
        self.first_layer_conv = tf.keras.layers.Conv2D(
            number_of_start_kernels,
            kernel_shape,
            activation=activation,
            padding='same',
            kernel_initializer=kernel_initializer)

        for k in range(unet_levels):
            self.down_blocks.append(
                DownLayer(number_of_start_kernels * (k + 1),
                          kernel_shape,
                          activation,
                          pooling_amount,
                          dropout_rate,
                          kernel_initializer=kernel_initializer))
        for k in reversed(range(unet_levels)):
            self.up_blocks.append(
                UpLayer(number_of_start_kernels * (k + 1),
                        kernel_shape,
                        activation,
                        pooling_amount,
                        dropout_rate,
                        kernel_initializer=kernel_initializer))

        self.output_layer = tf.keras.layers.Conv2D(
            number_of_categories,
            1,
            activation='softmax' if number_of_categories > 1 else 'sigmoid',
            padding='same',
            kernel_initializer=kernel_initializer)

    def call(self, input_tensor, training=False):
        down_outputs = []

        x = self.first_layer_conv(input_tensor)
        down_outputs.append(x)
        for k in range(0, self.unet_levels):
            x = self.down_blocks[k](x, training)
            down_outputs.append(x)
        down_outputs = down_outputs[::-1]

        x = self.up_blocks[0](down_outputs[0], down_outputs[1], training)
        for k in range(1, self.unet_levels):
            x = self.up_blocks[k](x, down_outputs[k + 1], training)
        return self.output_layer(x)
