# Building and Road Network Segmentation of Satellite Images 

[![Python application](https://github.com/Luke-Pratley/building_road_segmentation/actions/workflows/python-app.yml/badge.svg)](https://github.com/Luke-Pratley/building_road_segmentation/actions/workflows/python-app.yml)

In this repository, I investigate ways to semantically segment buildings and roads in top down. The motivation is that humanitarian and natural disasters can impact roads and buildings, and having a tool that can quickly survey them is useful. I make use of tensorflow to create a UNet and perform the prediction.

## Example

Below we display an example of an input image and the corresponding ground truth and predicted road network and building footprints.
<div style="text-align:center"><img src="https://raw.githubusercontent.com/Luke-Pratley/building_road_segmentation/getting_ready_for_submission/Vegas_input.png" />
<img src="https://raw.githubusercontent.com/Luke-Pratley/building_road_segmentation/getting_ready_for_submission/Vegas_output.png" /></div>
## The Data

The training and test data is taken from two SpaceNet competitions [2 and 3](https://spacenet.ai/spacenet-buildings-dataset-v2/), where the road and building data are cross matched. The foler `data_cleaning_EDA` holds the notebooks that cover the EDA and data cleaning process.

## The Models

There are a number of models implemented for testing in this repository, all of which are variations of UNet:
- Original UNet ([U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597))
- Residual UNet ([A deep Residual U-Net convolutional neural network for automated lung segmentation in computed tomography images, Khanna et al. (2021)](https://www.sciencedirect.com/science/article/abs/pii/S0208521620300887))
- Attention UNet ([Attention U-Net: Learning Where to Look for the Pancreas, Oktay et al. (2018)](https://arxiv.org/abs/1804.03999))
- Residual Attention UNet ([Residual Attention U-Net for Semantic Segmentation of Cataract Surgical Instruments, Zhen-Liang et al.(2019)](https://arxiv.org/abs/1909.10360))
- EfficientNet UNet ([Eff-UNet: A Novel Architecture for Semantic Segmentation in Unstructured
Environment, Baheti et al. (2020)](https://ieeexplore.ieee.org/document/9150621))


## Requirements

For a good list of requirements, you can check the testing build in `.github/workflows/python-app.yml`. 

It should work with Python 3.8 or greater. 

Note: raterio and gdal can have issues on windows. The order of import in python of geopandas and rasterio can be important.

For the EDA: 
- geopandas 
- pandas
- matplotlib
- rasterio
- numpy
- gdal

For the Training/Inference:
- PIL/Pillow
- Tensorflow 2.8
 
