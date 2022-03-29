#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import building_road_segmentation.unet_factory as unet_factory
import building_road_segmentation.data_generator as data_generator
import building_road_segmentation.loss_functions as loss_functions


def test_docstrings():
    # Testing that unet_factory has docstrings
    for name, val in unet_factory.__dict__.items():
        assert (val.__doc__ != ""), f'{name} has no docstring.'
    # Testing that data_generator has docstrings
    for name, val in data_generator.__dict__.items():
        assert (val.__doc__ != ""), f'{name} has no docstring.'
    # Testing that unet_factory has docstrings
    for name, val in loss_functions.__dict__.items():
        assert (val.__doc__ != ""), f'{name} has no docstring.'
