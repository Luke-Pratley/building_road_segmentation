#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os

import numpy as np
import rasterio
import geopandas as gpd
import rasterio.features
from rasterio.merge import merge
from rasterio.plot import show
import matplotlib.pyplot as plt


def get_directories_dictionary(data_directory, dataset_index):
    """ 
    Returns a dictionary of directories within the data directory.
    
    inputs:
    data_directory - path to the data 
    data_index - chooses the dataset in the data_directory to create a dictionary

    outputs:
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files.

    """
    assert isinstance(dataset_index, int), "data index must be an integer"
    assert isinstance(data_directory, str), "data_directory must be a string"
    datasets = glob.glob(data_directory + '*')
    assert len(
        data_directory) > 0, "There are no folders in the dataset directory."
    assert len(data_directory) > dataset_index, "data index is out of range."
    datasets = [d.split('\\')[-1] for d in datasets]
    directories = glob.glob(data_directory + f'{datasets[dataset_index]}\\*')
    directories.sort()
    directories_dict = dict()
    for d in directories:
        key = d.split('\\')[-1]
        if key != 'summaryData':
            directories_dict[key] = d
    return directories_dict


def get_image_path(directories_dict, folder, image_name):
    """
    Returns a path to the .tif file given the folder name and the image name
    
    inputs:
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files._name
    folder - folder name (string)
    image_name - image name (string)
    
    output:
    path to the .tif file
    
    """
    assert isinstance(folder, str), "Folder name must be a string"
    assert isinstance(image_name, str), "Image name must be a string"
    assert isinstance(directories_dict,
                      dict), "You did not pass a dictionary of directories"
    path = f'{directories_dict[folder]}\\{folder}_{image_name}.tif'
    assert os.path.exists(path), f'{path} does not exist'
    return path


def get_building_mask_path(directories_dict, im):
    """
    get the path to the building mask given the image ID

    inputs:
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files.
    im - image name (string)

    output:
    return the numpy array

    """
    assert isinstance(im, str), "Image name must be a string"
    assert isinstance(directories_dict,
                      dict), "You did not pass a dictionary of directories"
    path = f'{directories_dict["building_mask"]}\\building_mask_{im}.npy'
    return path


def get_road_mask_path(directories_dict, im):
    """
    get the path to the road mask given the image ID

    inputs:
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files.
    im - image name (string)

    output:
    return the numpy array

    """
    assert isinstance(im, str), "Image name must be a string"
    assert isinstance(directories_dict,
                      dict), "You did not pass a dictionary of directories"
    path = f'{directories_dict["road_mask"]}\\road_mask_{im}.npy'
    return path


def get_geopandas_for_image(directories_dict, im):
    """
    Returns a geopandas dataframe for a given image name

    inputs:
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files.
    im - image name (string)

    output:
    geopandas dataframe

    """
    assert isinstance(im, str), "Image name must be a string"
    assert isinstance(directories_dict,
                      dict), "You did not pass a dictionary of directories"
    path = f'{directories_dict["geojson"]}\\buildings\\buildings_{im}.geojson'
    assert os.path.exists(path), f'{path} does not exist'
    return gpd.read_file(path)


def plot_image(directories_dict, image_name, figsize=(10, 10)):
    """
    Plots images and building locations for a given image name.

    input:
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files.
    image_name: The name of the image to plot.

    output:
    None
    """
    assert isinstance(image_name, str), "Image name must be a string"
    assert isinstance(directories_dict,
                      dict), "Directories dict must be a dictionary"
    plt.rcParams.update({'font.size': 14})
    gpd_buildings_df = get_geopandas_for_image(directories_dict, image_name)
    fig, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=figsize)
    with rasterio.open(
            get_image_path(directories_dict, 'RGB-PanSharpen',
                           image_name)) as image:
        show(image.read() / image.read().max(),
             ax=ax,
             adjust=True,
             transform=image.transform,
             title=f'RGB-PanSharpen')
    gpd_buildings_df.plot(ax=ax, facecolor='red', edgecolor='red', alpha=0.5)
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel('Lon.')
    ax.set_ylabel('Lat.')
    fig.tight_layout()
    plt.show()


def plot_images(directories_dict, image_name):
    """
    Plots images and building locations for a given image name.

    input:
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files.
    image_name: The name of the image to plot.

    output:
    None
    """
    assert isinstance(image_name, str), "Image name must be a string"
    assert isinstance(directories_dict,
                      dict), "Directories dict must be a dictionary"
    plt.rcParams.update({'font.size': 14})
    gpd_buildings_df = get_geopandas_for_image(directories_dict, image_name)
    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 10))
    with rasterio.open(get_image_path(directories_dict, 'MUL',
                                      image_name)) as image:
        show(image, ax=ax[0, 0], title=f'MUL')
    gpd_buildings_df.plot(ax=ax[0, 0],
                          facecolor='red',
                          edgecolor='red',
                          alpha=0.5)
    with rasterio.open(
            get_image_path(directories_dict, 'MUL-PanSharpen',
                           image_name)) as image:
        show(image, ax=ax[0, 1], title=f'MUL-PanSharpen')
    gpd_buildings_df.plot(ax=ax[0, 1],
                          facecolor='red',
                          edgecolor='red',
                          alpha=0.5)
    with rasterio.open(get_image_path(directories_dict, 'PAN',
                                      image_name)) as image:
        show(image, ax=ax[1, 0], title=f'PAN')
    gpd_buildings_df.plot(ax=ax[1, 0],
                          facecolor='red',
                          edgecolor='red',
                          alpha=0.5)
    with rasterio.open(
            get_image_path(directories_dict, 'RGB-PanSharpen',
                           image_name)) as image:
        show(image.read() / image.read().max(),
             ax=ax[1, 1],
             adjust=True,
             transform=image.transform,
             title=f'RGB-PanSharpen')
    gpd_buildings_df.plot(ax=ax[1, 1],
                          facecolor='red',
                          edgecolor='red',
                          alpha=0.5)
    [k.ticklabel_format(useOffset=False) for k in ax.ravel()]
    ax[1, 0].set_xlabel('Lon.')
    ax[1, 1].set_xlabel('Lon.')
    ax[0, 0].set_ylabel('Lat.')
    ax[1, 0].set_ylabel('Lat.')
    fig.tight_layout()
    plt.show()


def create_mask(directories_dict, image_name):
    """
    Input
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files.
    image_name - this is the image ID to create the mask for
    
    Output - this is the boolean valued mask where the buildings are
    """
    assert isinstance(image_name, str), "Image name must be a string"
    with rasterio.open(
            get_image_path(directories_dict, 'RGB-PanSharpen',
                           image_name)) as image:
        mask = rasterio.features.geometry_mask(get_geopandas_for_image(
            directories_dict, image_name)['geometry'],
                                               image.shape,
                                               image.transform,
                                               all_touched=False,
                                               invert=True)
    return mask


def load_building_mask(directories_dict, image_name):
    """
    Loads a building mask.

    Input
    directories_dict - a dictionary that contains the dictories of the images, masks, and geojson files.
    image_name - this is the image ID to create the mask for
    
    Output - this is the boolean valued mask where the buildings are
    """
    assert isinstance(image_name, str), "Image name must be a string"
    assert isinstance(directories_dict,
                      dict), "Directories dict must be a dictionary"
    mask = np.load(get_building_mask_path(directories_dict, image_name))
    assert mask.dtype == bool, "Mask is not a bool"
    return mask

