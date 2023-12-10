### Project File Description

- data_augmentation.ipynb - Code to create the final audio dataset from the ASVspoof2019 dataset
- image_transformation.py - Script to convert audio files to various spectrogram images
- waveform_to_image_convert.py - Script for converting audio files to waveform images
- datastore.m - MATLAB script to load the image datasets into a MATLAB workspace to use the deepNetworkDesigner
- evaluate.m - MATLAB script to evaluate the trained ResNet-50 & SqueezeNet models on the test set for the specific spectrogram type
