# Image Annotation and Dataset Creation Pipeline
This repository contains a pipeline for creating annotations for images and generating a dataset from these annotations. It also includes tests for the models used in the pipeline.

## Overview
The pipeline.py script is responsible for the entire process, which includes creating annotations for an image and generating a dataset from these annotations. It is designed to be flexible and scalable, allowing for easy adjustments and improvements.

The test_models.py script contains unit tests for the models used in the pipeline. These tests ensure that the models are working as expected and help maintain the quality and reliability of the predictions.

In the test_models.py script, several machine learning models are trained and tested on a dataset. The models include Decision Tree, Logistic Regression, Neural Network, K-Nearest Neighbors, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost. The script loads the dataset, splits it into training and testing sets, trains each model on the training set, and then tests each model on the testing set. The performance of each model is saved to a JSON file.

## Installation
To install the necessary dependencies, run the following command:
```
pip install -r requirements.txt
```

## Usage
To run the pipeline, use the following command:
```
python pipeline.py
```

To run the tests for the models, use the following command:
```
python test_models.py
```