# Breast Cancer Classification using Machine Learning

This repository contains code and analysis for predicting breast cancer types (malignant or benign) using machine learning models. It includes data loading, preprocessing, visualization, model building, hyperparameter tuning, and model evaluation.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Data](#data)
5. [Models](#models)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)

## Project Overview

This project focuses on classifying breast cancer diagnoses as malignant or benign using machine learning algorithms. The data used comes from the Breast Cancer Wisconsin dataset, which contains several features derived from images of fine needle aspirates (FNA) of breast masses. These features are used to train machine learning models to predict the diagnosis based on the characteristics of the cell nuclei.

The project involves:
- Data exploration and visualization
- Data preprocessing and feature selection
- Training machine learning models
- Hyperparameter tuning for optimization
- Evaluating model performance using metrics like accuracy, precision, recall, and F1 score

## Features

The dataset contains the following key features for each sample:

- `radius_mean`: Mean of distances from the center to points on the perimeter
- `texture_mean`: Standard deviation of gray-scale values
- `perimeter_mean`: Perimeter of the tumor
- `area_mean`: Area of the tumor
- `smoothness_mean`: Local variation in radius lengths
- `compactness_mean`: Perimeter² / Area - 1.0
- `concavity_mean`: Severity of concave portions of the contour
- `symmetry_mean`: Symmetry of the tumor
- `fractal_dimension_mean`: "Coastline approximation" - 1D measure of complexity
- Plus many more features related to worst-case and mean values of these measurements.

## Installation

### Requirements
To run this project locally, you will need to have Python installed along with a few libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the necessary packages using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

### Data
The dataset used in this project is the Breast Cancer Wisconsin Diagnostic Dataset. It contains 569 samples with 30 features related to cell nuclei measurements obtained from digital images of a breast mass biopsy.

Target: Diagnosis (M for malignant, B for benign)
Source: UCI Machine Learning Repository

### Models
The following machine learning models were implemented and compared in this project:

- Support Vector Machine (SVM): A supervised learning model that uses hyperplanes to separate data points.
- K-Nearest Neighbors (KNN): A classification algorithm that predicts the class of a sample based on its k-nearest neighbors.
- Decision Trees: A flowchart-like tree structure where an internal node represents a feature and each leaf node represents an outcome.
- Random Forest: An ensemble method that creates a forest of decision trees to improve accuracy.

### Hyperparameter Tuning
Hyperparameter tuning was performed using techniques like Grid Search and Randomized Search to optimize model performance.

- Grid Search: Tests a specified range of hyperparameters to find the best combination.
- Randomized Search: Randomly searches through a distribution of hyperparameter values.

### Evaluation Metrics
Several metrics were used to evaluate the performance of the models:

- Accuracy: Proportion of correctly classified instances.
- Precision: Proportion of true positive predictions among all positive predictions.
- Recall: Proportion of true positive predictions among all actual positives.
- F1 Score: Harmonic mean of precision and recall, providing a balanced metric.

### Results
After training the models and tuning hyperparameters, the performance of each model was evaluated. The Support Vector Machine (SVM) provided the highest accuracy and F1 score on the test set. Below is a summary of the results:

| Model        | Accuracy | Precision | Recall | F1 Score |
|--------------|----------|-----------|--------|----------|
| SVM          | 97%      | 0.98      | 0.96   | 0.97     |
| KNN          | 95%      | 0.96      | 0.94   | 0.95     |
| Decision Tree| 93%      | 0.92      | 0.91   | 0.92     |


