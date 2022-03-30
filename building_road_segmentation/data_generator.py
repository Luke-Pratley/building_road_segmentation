#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
import tensorflow as tf
import math
import tensorflow.keras.utils
import numpy as np
import PIL
import tensorflow.image


def basic_asserts(x_set, y_set, batch_size):
    """
    A function that asserts the inputs for the data generators.

    Input:
        x_set: List of image paths.
        y_set: List of label paths.
        batch_size: The size of each training batch.
    """
    assert batch_size > 0, "Batch size must be larger than zero."
    assert isinstance(
        x_set,
        (list, np.array)), "list of paths must be a python list or np.array."
    assert isinstance(
        y_set,
        (list, np.array)), "list of paths must be a python list or np.array."
    assert all(isinstance(k, str)
               for k in x_set), "list of paths must be strings."
    assert all(isinstance(k, str)
               for k in y_set), "list of paths must be strings"
    assert len(x_set) == len(
        y_set), "The number of inputs and targets must be the same."
    assert len(x_set) > 0, "There must be more than zero inputs."


class READ_DATA(tf.keras.utils.Sequence):
    """
    A generator that reads the images and labels from file paths 
        into batches of images.
    """

    def __init__(self, x_set, y_set, batch_size):
        """
        The constructor for the data generator.

        Input:
            x_set: a list of file paths to images saved as .png.
            y_set: a list of file paths to images saved as .npy.
            batch_size: an integer for the batch size to use during training.
        """
        # assert input values
        basic_asserts(x_set, y_set, batch_size)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        """
        Calculates the total number of batches.

        Output:
            size: An integer that calculates the number of batches.
        """
        size = math.ceil(len(self.x) / self.batch_size)
        assert size > 0, (
            "The total number of batches needs to be greater than zero.")
        return size

    def __getitem__(self, idx):
        """ 
        Output the batch of features and labels given the batch number.

        Input:
            idx: The batch number as an integer.

        Output:
            numpy_array_of_images: a numpy array of images of size 
            (batch_size, imsize_y, imsize_x, 3), values between 0 and 1.
            numpy_array_of_labels: a numpy array of images of size 
            (batch_size, imsize_y, imsize_x, class_num), values are 0 or 1.
        """
        assert idx >= 0, "batch index must be greater than zero."
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(
            [np.array(PIL.Image.open(im)) / 255. for im in batch_x],
            dtype=np.float32), np.array([
                np.load(file_name).astype(np.float32) for file_name in batch_y
            ])


class READ_AND_AUGMENT_DATA(tf.keras.utils.Sequence):
    """
    A generator that reads the images and labels from file paths 
        into batches of images with selected augmentations and transforms.
    """

    class Augment(Enum):
        """
        An enum class that lists the types of augmentations to perform on 
        input and target images.
        """
        ORIGINAL = 1  # identity
        LRFLIP = 2  # left right flip
        UDFLIP = 3  # up down flip
        ROT90 = 4  # rotate 90 degrees
        ROT270 = 5  # rotate -90 degrees

    def __init__(self, x_set, y_set, batch_size):
        """
        The constructor for the data generator with augmentations.

        Input:
            x_set: a list of file paths to images saved as .png.
            y_set: a list of file paths to images saved as .npy.
            batch_size: an integer for the batch size to use during training.
        """
        # assert input values
        basic_asserts(x_set, y_set, batch_size)
        self.batch_size = batch_size
        self.x = []
        self.y = []
        for aug in self.Augment:
            self.x += [(k, aug) for k in x_set]
            self.y += [(k, aug) for k in y_set]

    def __len__(self):
        """
        Calculates the total number of batches.

        Output:
            size: An integer that calculates the number of batches.
        """
        size = math.ceil(len(self.x) / self.batch_size)
        assert size > 0, (
            "The total number of batches needs to be greater than zero.")
        return size

    def apply_augment(self, image, aug):
        """
        A function that returns a transformed image given an agumentation.

        Input:
            image: an input image.
            aug: an augmentation in the Augment enum class.

        Output:
            image: The transformed image.
        """
        if aug is self.Augment.ORIGINAL:
            return image
        if aug is self.Augment.LRFLIP:
            return tf.image.flip_left_right(image)
        if aug is self.Augment.UDFLIP:
            return tf.image.flip_up_down(image)
        if aug is self.Augment.ROT90:
            return tf.image.rot90(image)
        if aug is self.Augment.ROT270:
            return tf.image.rot90(image, k=3)
        RuntimeError("Augment not recognized.")

    def __getitem__(self, idx):
        """ 
        Output the batch of augmented features 
        and labels given the batch number.

        Input:
            idx: The batch number as an integer.

        Output:
            numpy_array_of_images: a numpy array of augmented images of size 
            (batch_size, imsize_y, imsize_x, 3), values between 0 and 1.
            numpy_array_of_labels: a numpy array of augmented images of size 
            (batch_size, imsize_y, imsize_x, class_num), values are 0 or 1.
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            self.apply_augment(np.array(PIL.Image.open(im[0])) / 255., im[1])
            for im in batch_x
        ],
                        dtype=np.float32), np.array([
                            self.apply_augment(
                                np.load(file_name[0]).astype(np.float32),
                                file_name[1]) for file_name in batch_y
                        ])


class TEST_DATA(tf.keras.utils.Sequence):
    """
    A basic data generator.
    """

    def __init__(self, x_set, y_set, batch_size):
        """
        The constructor for the data generator with lists of data.

        Input:
            x_set: a list of input objects.
            y_set: a list of target objects.
            batch_size: an integer for the batch size to use during training.
        """
        assert batch_size > 0, "batch size must be greater than zero."
        assert len(x_set) == len(
            y_set), "number of inputs and targets must be the same."
        assert len(x_set) > 0, "There must be more than zero inputs."
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        """
        Calculates the total number of batches.

        Output:
            size: An integer that calculates the number of batches.
        """
        size = math.ceil(len(self.x) / self.batch_size)
        assert size > 0, (
            "The total number of batches needs to be greater than zero.")
        return size

    def __getitem__(self, idx):
        """ 
        Output the batch of features and labels given the batch number.

        Input:
            idx: The batch number as an integer.

        Output:
            features: a list of inputs for batch idx.
            labels: a list of labels for batch idx.
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
