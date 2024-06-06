### Multi Author Writing Style Analysis 2024

This repository contains code and data used for the final project in the class "LP II" for the MSc in IT and Cognition at the University of Copenhagen. The project focussed on style change detection, specifically identifying author changes at the paragraph level. The task is derived from the PAN at CLEF 2024 challenge (see: https://pan.webis.de/clef24/pan24-web/style-change-detection.html). 

#### baseline
This folder contains a script to train a baseline CountVectorizer model using bag of words (BOW).

#### create_prediction_files
This folder contains a script to produce solution files in the output required by PAN at CLEF. 

#### data
This folder contains .csv files for the test and validation data and a script for the test split.

#### data_augmentation
This folder contains a script to create additional training data and balancing classes by oversampling. Additionally, it contains teh balanced and augmented training and validation datasets.

#### data_pipeline
This folder contains a script to process the original data provided by PAN at CLEF and the new training and validation datasets.

#### pan24-multi-author-analysis
Contains the original data provided by PAN at CLEF.

#### test-results
This folder contains the test results obtained from the frozen models. 

#### train-embeddings-model
This folder contains scripts to train and test the frozen embedding models.
