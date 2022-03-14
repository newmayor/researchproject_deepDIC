# Deep DIC: Deep Learning-Based Digital Image Correlation for End-to-End Displacement and Strain Measurement

Digital image correlation (DIC) has become an industry standard to retrieve accurate displacement and strain measurement in tensile testing and other material characterization using computer vision and image datasets. This project proposes a new deep learning-based DIC approach – Deep DIC, in which two convolutional neural networks, DisplacementNet and StrainNet, are designed to work together for end-to-end prediction of displacements and strains. 

A new dataset generation method is developed to synthesize a realistic and comprehensive dataset, including generation of speckle patterns and deformation of the speckle image with synthetic displacement field. 

The manuscript of this research work can be found in https://arxiv.org/abs/2110.13720

## Dependencies
Deep-DIC is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install the dependencies using the `pip install requirements.txt` . 

- Python 3.7 
- PyTorch (version = 1.16)
- Torchvision (version = 0.7.0)
- Pillow (version = 7.2)
- Scikit-learn==1.0.1
- numpy
- scipy
- CUDA

## Overview
We provide:
- Datasets: training dataset, validation dataset and test dataset.
      https://drive.google.com/drive/folders/1n2axHsJ3flHxk_edceY6eOfiX7GjW_d6?usp=sharing
- Pre-trained models:
      https://drive.google.com/drive/folders/1n2axHsJ3flHxk_edceY6eOfiX7GjW_d6?usp=sharing
    - DisplacementNet
    - StrainNet
- Code to test with pair of speckle images.
- Code to train a the two CNNs with dataset.
