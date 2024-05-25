[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LkCf-P6F)
ConfigWithYourProjectName
==============================
# DSEI210-S24-Final-Project

# Cat and Dog Image Classification Project

## Overview
This project focuses on building a robust model for classifying images of cats and dogs. It employs various machine learning and deep learning techniques, ranging from traditional algorithms to advanced techiniques as CNN and ResNet50

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Descriptions](#model-descriptions)
- [Results](#results)
- [Contributors](#contributors)
- [Bibliography](#bibliography)

## Project Structure
- `notebooks/`: Jupyter notebooks with detailed explanations and code for all the different methods.
- `models/`: Source code for the models and data processing scripts.
- `reports/`: Contains project reports and visualizations.
- `README.md`: This file.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cat-dog-classification.git
   cd cat-dog-classification

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`

3. Istall the required packeges:
    ```bash
    pip install -r requirements.txt


## Usage 
Run the Jupyter notebooks to explore the data and train models.
Use the provided scripts in the src/ directory to preprocess data and train models from the command line.

## Model Description

**HOG + SVM**
Histogram of Oriented Gradients (HOG) features are extracted and fed into a Support Vector Machine (SVM) for classification. This method achieved an accuracy of 78%.

**Random Forest**
A Random Forest classifier is used with pixel values as features, resulting in a lower accuracy compared to HOG + SVM.

**Bag of Visual Words (BOVW)**
Uses a vocabulary of BRISK features to classify images with a K-Nearest Neighbors (KNN) classifier. The accuracy achieved was 67.5%.

**Convolutional Neural Networks (CNN)**
A CNN was implemented using TensorFlow and Keras, achieving high accuracy by learning hierarchical representations of the images.

**ResNet-50**
Utilizes the ResNet-50 architecture pre-trained on ImageNet, fine-tuned for the cat and dog classification task. This model provided the highest accuracy.

## Results
* HOG + SVM: 78% accuracy
* Random Forest: Lower accuracy
* BOVW: 67.5% accuracy
* CNN: High accuracy with advanced fitting and regularization techniques
* ResNet-50: Highest accuracy among all models

## Contributors

* Alexander Sandoval
* Ryan Goldberg
* Blanche Horbach
* Valentina Samboni

## Bibliography
Raschka, Sebastian, et al. Machine Learning with PyTorch and Scikit-Learn: Develop Machine Learning and Deep Learning Models with Python. Packt, 2022.
Shalev-Shwartz, Shai, and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014.
Hastie, Trevor, et al. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. 2nd ed. Springer, 2009.


------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third-party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. The naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
