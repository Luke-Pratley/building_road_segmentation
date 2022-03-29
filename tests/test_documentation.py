#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from inspect import getmembers, isfunction
import building_road_segmentation.unet_factory as unet_factory
import building_road_segmentation.optimization_factory as optimization_factory
import building_road_segmentation.utilities as utilities
import building_road_segmentation.data_generator as data_generator
import building_road_segmentation.loss_functions as loss_functions


def test_docstrings():
    for module in [
            unet_factory, optimization_factory, utilities, data_generator,
            loss_functions
    ]:
        for name, val in getmembers(module, isfunction):
            print(name)
            assert isinstance(val.__doc__, str), f'{name} has no docstring'
            assert (val.__doc__ != ""), f'{name} has no docstring.'
