# Image Annotation and Dataset Creation Pipeline
This repository contains a pipeline for creating annotations for images and generating a dataset from these annotations. It also includes tests for the models used in the pipeline. Work in progress...

## Overview
The goal of this repository is to evaluate how well the method of converting an image to features using gray-level co-occurrence matrix (GLCM) analysis allows for the identification and differentiation of textures. This method analyzes texture properties such as contrast, dissimilarity, homogeneity, energy, and correlation from the input image. These properties are then used to generate features for the dataset, which are crucial for training and testing the machine learning models.

The pipeline.py script is responsible for the entire process, which includes creating annotations for an image and generating a dataset from these annotations. It is designed to be flexible and scalable, allowing for easy adjustments and improvements.

The train_models.py script contains unit tests for the models used in the pipeline. These tests ensure that the models are working as expected and help maintain the quality and reliability of the predictions.

In the test_models.py script, several machine learning models are trained and tested on a dataset. The models include Decision Tree, Logistic Regression, Neural Network, K-Nearest Neighbors, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost. The script loads the dataset, splits it into training and testing sets, trains each model on the training set, and then tests each model on the testing set. The performance of each model is saved to a JSON file.

### Texture Features from GLCM

Several statistical measures can be derived from the GLCM to quantify the texture of the image. Common features include:

**Contrast**:

Measures the local variations in the gray-level co-occurrence matrix.

$$ \text{Contrast} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} P(i, j) (i - j)^2 $$

**Dissimilarity**:

Similar to contrast but uses the absolute difference.

$$ \text{Dissimilarity} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} P(i, j) |i - j| $$

**Homogeneity**:

Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal.

$$ \text{Homogeneity} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} \frac{P(i, j)}{1 + (i - j)^2} $$

**Energy**:

The sum of squared elements in the GLCM. Also known as Angular Second Moment (ASM).

$$ \text{Energy} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} P(i, j)^2 $$

**Correlation**:

Measures how correlated a pixel is to its neighbor over the entire image.

$$ \text{Correlation} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} \frac{(i - \mu_i)(j - \mu_j) P(i, j)}{\sigma_i \sigma_j} $$

Where $\mu_i$ and $\mu_j$ are the means of $i$ and $j$ respectively, and $\sigma_i$ and $\sigma_j$ are the standard deviations of $i$ and $j$ respectively.



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
