# DED-IM: A Novel Method for Mapping and Path Planning in Wire Arc Directed Energy Deposition

![screenshot](imageabstract.png)

## Authors

[Matheus Antunes Chipanski](https://github.com/machipanski)

[Rodrigo Minetto](https://github.com/rminetto)

# DED-IM
This project is a novel image-based mapping and tool-path planning method for Wire Arc Directed Energy Deposition (WA-DED). The method automatically segments 3D models into distinct regions such as thin walls, contours, bottlenecks, and large internal areas by processing binary images. These segments enable precise, geometry-tailored tool-path generation, significantly improving control over material deposition and reducing arc interruptions. Key innovations include the application of medial axis transforms to accurately map thin walls, the introduction of oscillatory paths for bottleneck areas, and an adaptive weaving pattern that minimizes voids while enhancing geometric precision. The method also integrates user-defined parameters, offering flexibility in region mapping and filling strategies to address specific manufacturing requirements. Experimental results demonstrate the efficacy of DED-IM, showing substantial reductions in common defects such as voids, insufficient material filling, and arc interruptions.

Developed in Python, the method takes a 3D model as input, along with machine-specific parameters, and generates G-code instructions that can be customized for different WA-DED machines. DED-IM automatically segments 3D models into regions like thin walls, contours, bottlenecks, and large internal areas by processing binary images. These parameters are fully adjustable, allowing users to tailor the tool-path strategy to their specific needs. The modular design also makes it easy to integrate new strategies, such as custom zigzag patterns. To use DED-IM in a production workflow, practitioners will need to adjust the software according to their machine specifications.

# Using the Program



## Example Models

There are single-layer simulations represented by single images into the  `<your-local-repository>/input` folder.
The 3D models used to test the algorithm are all present into the  `<your-local-repository>/input/stl_models` folder.

**Atention:** Due to the stert of the process relying on a older version of the `slicing with images` project, some of the models can be rotated to diferent positions, so it may be necessary to save the models in different orientations before generating the desired slices.


# Show the CNN output

You can also generate images from trained models. For this purpose, access the `src/utils/cnn` folder and find the `generate_inference.py` script. This script will pass the image patch through the selected model and generate a PNG image with the prediction. You need to define the `IMAGE_NAME` constant with the desired image, and set the `IMAGE_PATH` constant with the path where this image can be found. It is important to define `MASK_ALGORITHM` with the approach you want to use, `N_CHANNELS` and `N_FILTERS` with the number of channels used and number of filters of the model. Make sure that the trained weights defined in `WEIGHTS_FILE` is consistent with the parameters defined. After that you can run:

```
python generate_inference.py
```

You may want to compare the output produced by the different architectures and also the masks. The script `generate_figures.py` applies the different networks trained on all patches from the image defined in the `IMAGE_NAME` constant. The output folder will have the outputs produced by the CNNs, the available masks and an image path with the combination of channels 7, 6 and 2 in a PNG format. This script can be run using:
```
python generate_figures.py
```

# Useful stuff

In `src/utils` you will find some scripts that may help you understand the images and masks you are working with. For example, you may want to know how many fire pixels are in a mask, for this you can use the `count_fire_pixels.py` script. You just define the `IMAGE_NAME` with the desired mask name and run:

```shell
python count_fire_pixels.py
```

Alternatively you can define a partial name in `IMAGE_NAME` and a pattern to be found in `PATCHES_PATTERN`, and count fire pixels from many patches.

You may need to find an image with at least some minimal amount of fire pixels, the script `masks_with_at_least_n_fire_pixels.py` prints on the screen the image path of the masks in the `MASK_PATH` directory that has more fire pixels than `NUM_PIXELS` defined in the script. After defining the path and the amount of pixels you can run:
```
python masks_with_at_least_n_fire_pixels.py
```

The masks have the value 1 where fire occurs and 0 otherwise, because of that the masks will not display "white" and "black" if open in an image viewing program. To help you see the mask you can use the script `transform_mask.py`, this script will convert the image to a PNG with white and black pixels. You just need to define the mask you want to convert in the `MASK_PATH` constant and run:

```shell
python transform_mask.py
```

The images available in the dataset are also difficult to view, as they are 10-channel images. You can convert them into a visible format, with the combination of bands 7, 6 and 2, using the `convert_patch_to_3channels_image.py` script. You just need to set the path to the desired image in the `IMAGE` constant and run:

```shell
python convert_patch_to_3channels_image.py
```

If you trained the models from scratch and want to use those weights to compare against the manual annotations, use the `copy_trained_weights_to_manual_annotations_evaluation.py` script to copy the trained weights to the directories used to evaluate the models against the manual annotations:

```shell
python copy_trained_weights_to_manual_annotations_evaluation.py
```

The evaluation processes will generate many txt files, you can use the `purge_logs.py` script to erase them:

```shell
python purge_logs.py
```

In order to save space, only fire masks are available. In our code, the masks without fire are generated on the fly when needed, but you may want to see all annotations. You can use the `generate_nonfire_masks.py` script to generate the missing masks. It will create one mask for each patch filled with zeroes, using the metadata from the original patch. If a mask already exists it will be ignored. To run it:

```shell
python generate_nonfire_masks.py
```

# Environment preparation

The scripts in this repository were done using Tensorflow (GPU) version 1.13, using Keras 2.2 version. Since there are some Nvidia-drivers/CUDA/TensorFlow specific installation details we don't provide any enviroment preparation file, but the main libriries used are:
```
tensorflow-gpu
keras
numpy
cv2
scikit-learn
pandas
geopandas
rasterio
```
If you want to use TensorFlow 2.X you can follow some hints in the [tensorflow-2.md](tensorflow-2.md) file.


# Citation

If you find our work useful for your research, please [cite our paper](https://www.sciencedirect.com/science/article/abs/pii/S092427162100160X):

```
@article{DEALMEIDAPEREIRA2021171,
title = {Active fire detection in Landsat-8 imagery: A large-scale dataset and a deep-learning study},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {178},
pages = {171-186},
year = {2021},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2021.06.002},
url = {https://www.sciencedirect.com/science/article/pii/S092427162100160X},
author = {Gabriel Henrique {de Almeida Pereira} and Andre Minoro Fusioka and Bogdan Tomoyuki Nassu and Rodrigo Minetto},
keywords = {Active fire detection, Active fire segmentation, Active fire dataset, Convolutional neural network, Landsat-8 imagery},
}
```

Or access the [preprinted version](https://arxiv.org/abs/2101.03409).


# License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


