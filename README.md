[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LkCf-P6F)
ConfigWithYourProjectName
==============================
# DSEI210-S24-Final-Project

# Cat-Dog Image Classifier

## Project Description

This Applied Machine Learning project is designed to classify images into two categories: cats and dogs. It leverages several machine learning techniques and image processing methods to preprocess the images, extract features, and classify them using various algorithms.

## File Structure
- `/content/cat`: Directory containing cat images.
- `/content/dog`: Directory containing dog images.

## Requirements

This project requires Python 3 and the following libraries:
- pandas
- numpy
- scikit-learn
- scikit-image
- opencv-python
- matplotlib

## Usage 

1. **Image Preprocessing**: Images are converted to grayscale, resized, and subjected to various image processing techniques to extract features.
2. **Feature Extraction**: Features are extracted using a Histogram of oriented Gradients (HOG), Prewwit operator, and morphological edge detection.
3. **Modeling Training**: Several models are trained using techniques like Support Vector Machines, Random Forest, and K-Nearest Neighbors, with hyperparameter tuning via grid search.
4. **Evaluation**: Models are evaluated based on accuracy, and ensemble methods are utilized to enhance prediction performance.

## Key Functions 

- `resize_image`: Resizes images to a uniform size.
- `greyscale`: Uniform color
- `hog_features`: Extracts HOG features from the resized images.
- `prewitt_operator`: Applies the Prewitt operator to detect edges.
- `morphological_edge_detection`: Uses morphological operations to detect edges.
- `best_pca`: Finds the optimal number of principal components.
- `best_svc`, `best_rf`, `best_knn`: Functions for finding the best parameters and performing grid search for SVC, Random Forest, and KNN respectively.

## Example Code
To process images and extract features:
```python
cat_dog_df['resized_image'] = cat_dog_df['image_path'].apply(lambda x: resize_image(x, size) if x else None)
cat_dog_df['hog_features'] = cat_dog_df['resized_image'].apply(lambda x: hog_features(x, pixels_per_cell, cells_per_block) if x is not None and x.shape else None)

## To train and evaluate models

best_svc(n_components, X_train, X_test, y_train, y_test)
scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
print(f'ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]')

```
## Visualization

The project includes code to plot the results of PCA and the feature extraction methods to visualize their effectiveness.

## Contributors

* Alexander Sandoval
* Ryan Goldberg
* Blanche Horbach
* Valentina Samboni 


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
