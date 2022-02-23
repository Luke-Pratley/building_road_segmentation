#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import building_road_segmentation


def test_path_missing():
    with pytest.raises(AssertionError):
        data_directory = ''
        dataset_index = 1
        #it will throw because there are no directories found
        result = building_road_segmentation.get_directories_dictionary(
            data_directory, dataset_index)

