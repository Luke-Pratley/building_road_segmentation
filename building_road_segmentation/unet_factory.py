#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import tensorflow.keras.applications.efficientnet


class ConvBlock(tf.keras.Model):
    """
    A key convolutional block for UNet: (conv2d -> batchnorm -> activation) x 2.
    """

    def __init__(self, number_of_start_kernels, kernel_shape, activation,
                 residual, kernel_initializer):
        """
        Initilizing the layers.

        Input:
            number_of_start_kernels: The number of convolution kernels
                                        to use in a convolutional layer
            kernel_shape: The size of a kernel.
            activation: The name or object for the activation function.
            residual: If to include a residual layer (for res-UNet) (bool)
            kernel_initializer: How to initialize the kernel weights.
        """
        super(ConvBlock, self).__init__(name='')
        self.residual = residual
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
        if self.residual:
            self.add = tf.keras.layers.Add()
            self.conv3 = tf.keras.layers.Conv2D(
                number_of_start_kernels,
                1,
                padding='same',
                kernel_initializer=kernel_initializer)

    def call(self, input_tensor, training=False):
        """
        A function that calls the layers.

        Input:
            input_tensor: input to the first layer.
            training: bool that is True if paramters are being trained.

        Output:
            output_tensor: The output tensor.
        """
        x = self.conv1(input_tensor)
        x = self.bn1(x, training)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.bn2(x, training)
        x = self.activation2(x)
        if self.residual:
            y = self.conv3(input_tensor)
            x = self.add([y, x])
        return x


class DownLayer(tf.keras.Model):
    """
    A UNet layer within the encoder that pools/downsamples the input and applies a ConvBlock.
    """

    def __init__(self, number_of_start_kernels, kernel_shape, activation,
                 pooling_amount, dropout_rate, residual, kernel_initializer):
        """
        Initilizing the layers.

        Input:
            number_of_start_kernels: The number of convolution kernels
                                        to use in a convolutional layer
            kernel_shape: The size of a kernel.
            activation: The name or object for the activation function.
            pooling_amount: The factor to downsample/pool by.
            dropout_rate: The dropout rate.
            residual: If to include a residual layer (for res-UNet) (bool)
            kernel_initializer: How to initialize the kernel weights.
        """
        super(DownLayer, self).__init__(name='')
        self.pool = tf.keras.layers.MaxPooling2D(
            (pooling_amount, pooling_amount))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.convblock = ConvBlock(number_of_start_kernels,
                                   kernel_shape,
                                   activation,
                                   residual,
                                   kernel_initializer=kernel_initializer)

    def call(self, input_tensor, training=False):
        """
        Calls the layers on the input, pooling -> dropout -> ConvBlock.

        Input:
            input_tensor: input to the first layer.
            training: bool that is True if paramters are being trained.

        Output:
            output_tensor: The output tensor.

        """
        x = self.pool(input_tensor)
        x = self.dropout(x, training)
        return self.convblock(x, training)


class UpLayer(tf.keras.Model):
    """
    A UNet layer within the encoder that upsamples the input while
    concatenating output from a DownLayer then applies
    a dropout and a ConvBlock layer.
    """

    def __init__(self, number_of_start_kernels, kernel_shape, activation,
                 pooling_amount, dropout_rate, residual, kernel_initializer):
        """
        Initilizing the layers.

        Input:
            number_of_start_kernels: The number of convolution kernels
                                        to use in a convolutional layer
            kernel_shape: The size of a kernel.
            activation: The name or object for the activation function.
            pooling_amount: The factor to downsample/pool by.
            dropout_rate: The dropout rate.
            residual: If to include a residual layer (for res-UNet) (bool)
            kernel_initializer: How to initialize the kernel weights.
        """
        super(UpLayer, self).__init__(name='')
        self.upsample = tf.keras.layers.UpSampling2D(size=(pooling_amount,
                                                           pooling_amount))
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.convblock = ConvBlock(number_of_start_kernels,
                                   kernel_shape,
                                   activation,
                                   residual,
                                   kernel_initializer=kernel_initializer)

    def call(self, input_tensor, training=False):
        """
        Calls the layers on the input, pooling -> dropout -> ConvBlock.

        Input:
            input_tensor: the input is actually two inputs.
            input[0] (layer to be upsampled) and input[1] (layer from DownLayer)
            training: bool that is True if paramters are being trained.

        Output:
            output_tensor: The output tensor.

        """
        x = self.upsample(input_tensor[0])
        y = input_tensor[1]
        x = self.concat([x, y])
        x = self.dropout(x, training)
        x = self.convblock(x, training)
        return x


class BasicUNet(tf.keras.Model):
    """
    A UNet Convolutional Neural Network.

    The network will take an input image (values between 0 and 1) 
    and perform semantic segmentation for a fixed number of classes.
    """

    def __init__(self,
                 number_of_categories,
                 unet_levels,
                 number_of_start_kernels,
                 kernel_shape,
                 activation,
                 final_activation,
                 pooling_amount,
                 dropout_rate,
                 residual,
                 kernel_initializer=tf.keras.initializers.he_normal()):
        """
        Initilizing the layers.

        Input:
            number_of_categories: The number of categories to classify.
            unet_levels: The number of convolutional layers for the encoder
                         and decoder respectively.
            number_of_start_kernels: The number of convolution kernels
                                        to use in a convolutional layer.
            kernel_shape: The size of a kernel.
            activation: The name or object for the activation function.
            final_activation: The activation for the output layer,
                              e.g., sigmoid or softmax for probability.
            pooling_amount: The factor to downsample/pool by.
            dropout_rate: The dropout rate.
            residual: If to include a residual layer (for res-UNet) (bool).
            kernel_initializer: How to initialize the kernel weights.
        """
        super(BasicUNet, self).__init__(name='')
        assert unet_levels > 0, "Unet levels is less than 1"
        assert number_of_categories > 0, "number of classes/categories less than 1"
        self.unet_levels = unet_levels
        self.number_of_start_kernels = number_of_start_kernels
        self.pooling_amount = pooling_amount
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
                          residual,
                          kernel_initializer=kernel_initializer))
        for k in reversed(range(unet_levels)):
            self.up_blocks.append(
                UpLayer(number_of_start_kernels * (k + 1),
                        kernel_shape,
                        activation,
                        pooling_amount,
                        dropout_rate,
                        residual,
                        kernel_initializer=kernel_initializer))

        self.output_layer = tf.keras.layers.Conv2D(
            number_of_categories,
            1,
            activation=final_activation,
            padding='same',
            kernel_initializer=kernel_initializer)

    def call(self, input_tensor, training=False):
        """
        Calls the layers on the input.

        Input:
            input_tensor: input image (batches, height, width, channels).
            training: bool that is True if paramters are being trained.

        Output:
            output_tensor: The output tensor (batches, height, width, categories).

        """
        down_outputs = []

        x = self.first_layer_conv(input_tensor)
        down_outputs.append(x)
        for k in range(0, self.unet_levels):
            x = self.down_blocks[k](x, training)
            down_outputs.append(x)
        down_outputs = down_outputs[::-1]

        x = self.up_blocks[0]([down_outputs[0], down_outputs[1]], training)
        for k in range(1, self.unet_levels):
            x = self.up_blocks[k]([x, down_outputs[k + 1]], training)
        return self.output_layer(x)


class AttentionGate(tf.keras.Model):
    """
    The Attention Gate Layer for Attention UNet.

    This design is taken from
    Ozan et al. "Attention U-Net: Learning Where to Look for the Pancreas"

    """

    def __init__(self, num_filters, pooling_amount, kernel_initializer):
        """
        Initilizing the layers.

        Input:
            num_filters: The number of convolution kernels
                            to use in the attention process.
            pooling_amount: The factor to downsample/pool by.
            kernel_initializer: How to initialize the kernel weights.
        """
        super(AttentionGate, self).__init__(name='')

        self.W_gating_signal = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=kernel_initializer)
        self.W_input = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            kernel_initializer=kernel_initializer)
        self.add = tf.keras.layers.Add()
        self.acitvation1 = tf.keras.layers.Activation('relu')
        self.Ψ = tf.keras.layers.Conv2D(filters=1,
                                        kernel_size=(1, 1),
                                        padding='same',
                                        kernel_initializer=kernel_initializer)
        self.activation2 = tf.keras.layers.Activation('sigmoid')
        self.resampler = tf.keras.layers.UpSampling2D(size=(pooling_amount,
                                                            pooling_amount))
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        """
        Performs the attention gating process.

        Input:
            inputs: There are two inputs inputs[0] and inputs[1].
            inputs[0]: The input to the layer.
            inputs[1]: The input used as the gated signal.

        Output:
            output_tensor: The output tensor.
        """
        x = self.W_input(inputs[0])
        g = self.W_gating_signal(inputs[1])
        x = self.add([x, g])
        x = self.acitvation1(x)
        x = self.Ψ(x)
        x = self.activation2(x)
        alpha = self.resampler(x)
        return self.multiply([inputs[0], alpha])


class AttentionUNet(BasicUNet):
    """
    An Attention UNet Convolutional Neural Network.

    The network will take an input image (values between 0 and 1) 
    and perform semantic segmentation for a fixed number of classes.
    """

    def __init__(self, attention_intermediate_dim=None, *args, **kwargs):
        """
        Initilizing the layers.

        Input:
            attention_intermediate_dim: the kernel size in the attention gate.

        See UNet for args and kwargs.
        """
        super(AttentionUNet, self).__init__(*args, **kwargs)
        self.attention_gates = []
        if attention_intermediate_dim is None:
            attention_intermediate_dim = [
                self.number_of_start_kernels * (k + 1)
                for k in range(self.unet_levels)
            ]
        for levels in range(self.unet_levels):
            self.attention_gates.append(
                AttentionGate(
                    attention_intermediate_dim[levels],
                    self.pooling_amount,
                    kernel_initializer=tf.keras.initializers.Constant(
                        value=0)))

    def call(self, input_tensor, training=False):
        """
        Calls the layers on the input.

        Input:
            input_tensor: input image (batches, height, width, channels).
            training: bool that is True if paramters are being trained.

        Output:
            output_tensor: The output tensor.

        """
        down_outputs = []

        x = self.first_layer_conv(input_tensor)
        down_outputs.append(x)
        for k in range(0, self.unet_levels):
            x = self.down_blocks[k](x, training)
            down_outputs.append(x)
        down_outputs = down_outputs[::-1]

        gated_output = self.attention_gates[0](
            [down_outputs[1], down_outputs[0]])
        x = self.up_blocks[0]([down_outputs[0], gated_output], training)
        for k in range(1, self.unet_levels):
            gated_output = self.attention_gates[k]([down_outputs[k + 1], x])
            x = self.up_blocks[k]([x, gated_output], training)
        return self.output_layer(x)


def make_efficientnet_spine(
        levels,
        efnet_model=tensorflow.keras.applications.efficientnet.EfficientNetB4):
    """
    Takes an EfficientNet model from keras and turns it into an encoder for UNet.

    The end of each block is chosen by the final addtion/reduction layers in EfficientNet.

    Input:
        levels: the number of unet levels to find the blocks to use as output for the encoder
        efnet_model: the function from keras that pulls the efficientnet model.
    Returns:
        encoder: a model that takes the imput image and outputs the 
                 encoded image at each level of the UNet.
    """
    assert levels < 6, "Too many levels for efficientnet"
    # We are choosing the imagenet weights for transfer learning
    efnet = efnet_model(weights='imagenet',
                        include_top=False,
                        input_shape=(None, None, 3))
    get_layers = []
    for k in range(1, levels):
        for j in range(0, 20):
            try:
                efnet.get_layer(f'block{k}{chr(98 + j)}_add')
            except:
                get_layers.append(
                    efnet.get_layer(f'block{k}{chr(98 + j - 1)}_add'))
                break
    return tf.keras.Model(inputs=[efnet.layers[0].input],
                          outputs=[g.output for g in get_layers])


class EfficientNetUNet(tf.keras.Model):
    """
    An UNet Convolutional Neural Network with EfficientNet Encoding.

    The network will take an input image (values between 0 and 1) 
    and perform semantic segmentation for a fixed number of classes.
    """

    def __init__(self,
                 efficientnet,
                 number_of_categories,
                 unet_levels,
                 number_of_start_kernels,
                 kernel_shape,
                 activation,
                 final_activation,
                 pooling_amount,
                 dropout_rate,
                 residual,
                 kernel_initializer=tf.keras.initializers.he_normal()):
        """
        Initilizing the layers

        Input:
            efficientnet: The EfficientNet constructor from keras.
            number_of_categories: The number of categories to classify.
            unet_levels: The number of convolutional layers for the encoder
                         and decoder respectively.
            number_of_start_kernels: The number of convolution kernels
                                        to use in a convolutional layer.
            kernel_shape: The size of a kernel.
            activation: The name or object for the activation function.
            final_activation: The activation for the output layer,
                              e.g., sigmoid or softmax for probability.
            pooling_amount: The factor to downsample/pool by.
            dropout_rate: The dropout rate.
            residual: If to include a residual layer (for res-UNet) (bool).
            kernel_initializer: How to initialize the kernel weights.
        """
        super(EfficientNetUNet, self).__init__(name='')
        assert unet_levels > 0, "Unet levels is less than 1"
        assert number_of_categories > 0, "number of classes/categories less than 1"
        self.unet_levels = unet_levels
        self.down_blocks = []
        self.up_blocks = []

        self.efficientnet_spine = make_efficientnet_spine(
            unet_levels, efficientnet)

        for k in reversed(range(unet_levels - 1)):
            self.up_blocks.append(
                UpLayer(number_of_start_kernels * (k + 1),
                        kernel_shape,
                        activation,
                        pooling_amount,
                        dropout_rate,
                        residual,
                        kernel_initializer=kernel_initializer))

        self.output_layer = tf.keras.layers.Conv2D(
            number_of_categories,
            1,
            activation=final_activation,
            padding='same',
            kernel_initializer=kernel_initializer)

    def call(self, input_tensor, training=False):
        """
        Calls the layers on the input.

        Input:
            input_tensor: input image (batches, height, width, channels).
            training: bool that is True if paramters are being trained.

        Output:
            output_tensor: The output tensor (batches, height, width, categories).

        """
        down_outputs = []

        outputs = self.efficientnet_spine(input_tensor * 255., training)
        down_outputs = [k for k in outputs]
        down_outputs.insert(0, input_tensor)
        down_outputs = down_outputs[::-1]
        x = self.up_blocks[0]([down_outputs[0], down_outputs[1]], training)
        for k in range(1, self.unet_levels - 1):
            x = self.up_blocks[k]([x, down_outputs[k + 1]], training)
        return self.output_layer(x)
