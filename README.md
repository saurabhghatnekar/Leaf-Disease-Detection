# Leaf-Disease-Detection
Detection of various leaf diseases using GLCM features and Gradient Boosting Classifier


## NOTE
1. The dataset is downloaded from this link : https://github.com/Deqm525/FlowerLeafDiseaseDataset
2. Badam images were collected manually using Raspberry Pi cam


## Useage:
1. Run feature-extraction.py with appropriate folder (rose,badam,sunflower). It will generate a xlsx file with folder name
   This file contains features of each image in the selected folder and its corresponding class.
2. Next, run the learn.py with appropriate xlsx filename. The code will print the result which contains accuracy with and without cross-validation and classification report.

