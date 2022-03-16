# Generating dataset of speckle images

#### Ensuring dataset is generated as expected is very important before training the model.

File structure

      .
      ├── ...
      ├── dataset_generation                             
      │     ├── generate_specklepattern.py                  # generate pairs of small & large deformation speckle patterns and their groundtruths
      └── ...




- User will be prompted for pairs of images N desired. Try generating N = 1 or 2 to ensure generation works.

- User will be prompted to specify if generating a training set or test-validation set. 
      Enter 'train' or 'test'. Note: DisplacementNet requires images present in both directories to begin training.
      Dataset saved to directories below

      .
      ├── ...
      ├── dataset_generation
      │     ├── dataset_samples                             
      │     │     ├── gts1                      # groundtruths for model training
      │     │     │     ├── test                # gt's for cross validation step  
      │     │     ├── imgs                      # image pairs for model training
      │     │     │     ├── test                # image pairs for cross validation step  
      └── ...


- User will be prompted to specify if 'cracking' in images is desired:
      Enter 'Y' or 'N'. For first pass at running code, enter 'N'

- User will be prompted to specify which datapoint label to start from. This is handy if previous generation failed out, otherwise enter 0.

Once image generation is complete, ensure the dataset_samples dir contain the correct number of data. These will be used to train the models.


#### After generation
Use test scripts in `../test` to do sanity check on dataset just generated.

