# Deep DIC (Deep Learning for Digital Image Correlation)
### A look at Deep Learning for predicting strain and displacement fields in material grains using computer vision and Digital Image Correlation

Digital image correlation (DIC) has become an industry standard to retrieve accurate displacement and strain measurement in tensile testing and other material characterization using computer vision and image datasets. This project proposes a new deep learning-based DIC approach – Deep DIC, in which two convolutional neural networks, DisplacementNet and StrainNet, are designed to work together for end-to-end prediction of displacements and strains. 

A new dataset generation method is developed to synthesize a realistic and comprehensive dataset, including generation of speckle patterns and deformation of the speckle image with synthetic displacement field. 

The manuscript of this research work can be found in https://arxiv.org/abs/2110.13720

## Dependencies
Deep-DIC is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install the dependencies using `pip install requirements.txt` . 

- Python 3.7 
- PyTorch (version = 1.16)
- Torchvision (version = 0.7.0)
- Pillow (version = 7.2)
- Scikit-learn==1.0.1
- numpy
- scipy
- CUDA


## File Structure
      .
      ├── notebooks                             # contains ML model for DisplacementNet and notebook for testing ML model
      ├── dataset_generation                    # scripts to generate speckle image dataset
      ├── notebooks_archive                     # contains previously developed ML models to sanity check against
      ├── dataset_samples                       # samples of synethtically generate images used in training
      ├── test                                  # test scripts and notebooks for sanity checking data
      ├── requirements.txt                      # install dependencies from this file
      └── README.md

#### Main Notebooks
      .
      ├── ...
      ├── notebooks                             
      │     ├── DisplacementNet_train.ipynb                 # train DisplacementNet model on synthetic image dataset
      │     ├── experiment_displacementNet.ipynb            # test performance of trained D-Net model and visualize against groundtruth
      └── ...

#### Image Dataset Generation
      .
      ├── ...
      ├── dataset_generation                             
      │     ├── generate_specklepattern.py                  # generate small deformation speckle patterns
      │     ├── warp_severe.py                              # generate large deformation speckle patterns
      └── ...

#### Sample Dataset
      .
      ├── ...
      ├── dataset_samples                             
      │     ├── gts1                      # groundtruths for model training
      │     │     ├── test                # contains gt for cross validation step  
      │     ├── imgs                      # image pairs for model training
      │     │     ├── test                # contains image pairs for cross validation step  
      └── ...

#### Testing
      .
      ├── ...
      ├── test                                  # test scripts and notebooks for sanity checking data
      │     ├── stats_analysis.ipynb            # compute statistical metrics of synthetic image dataset
      │     ├── log_visualize.ipynb             # visualize model training loss
      │     ├── tensile_images.ipynb            # compile dataset of real tensile test images
      │     ├── experiment_tensileTest.ipynb    # test prediction of D-Net on real tensile test images
      └── ...


## Usage
