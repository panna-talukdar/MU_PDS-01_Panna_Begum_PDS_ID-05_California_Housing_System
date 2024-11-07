# California_Housing_Price_Prediction

This project uses a regression model to predict housing prices in California using the California Housing Dataset from Scikit-Learn. The model is developed in Python with libraries such as Scikit-Learn, Pandas, Numpy, Matplotlib, and Seaborn for data analysis, model training, and visualization.

## Project Overview

The objective of this project is to predict housing prices in California based on various socioeconomic and geographical features. This model can provide insights into the factors that influence housing prices.

## Dataset

The California Housing Dataset is a popular dataset included in Scikit-Learn, containing information on housing blocks in California. The features include median income, average rooms, house age, and more, with the target being the median house price.

## Requirements

- Python 3.x
- Libraries: Scikit-Learn, Pandas, Numpy, Matplotlib, Seaborn



## Installation

1.Clone the repository.

bash
  https://github.com/panna-talukdar/MU_PDS-01_Panna_Begum_PDS_ID-05_California_Housing_System.git

2.Install the required libraries.
bash
pip install -r requirements.txt


## Usage
1.Load and explore the data.

2.Preprocess data, scale features, and split into training and testing sets.

3.Train a regression model to predict housing prices.

4.Evaluate model performance using metrics like Mean Squared Error and R² score.

5.Visualize the results.

## Run the main script to execute the entire pipeline:
bash
python main.py


## Running on Google Colab

You can also run this project on Google Colab for a cloud-based, no-installation setup. Follow these steps:

1.Upload the project files to Google Drive or download them directly within Colab.

2.Open a new Colab notebook.
3.Install any required libraries by running:
bash
!pip install -r requirements.txt


4.Load the California Housing Dataset directly from Scikit-Learn in your Colab notebook with:
python
bash
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()


5.Follow the project steps, running each code block in individual cells as needed.
bash
from google.colab import drive
drive.mount('/content/drive')


6.Run the cells sequentially to complete the steps.

## Steps
### 1.Setup and Imports
Import libraries required for data handling, model training, evaluation, and visualization.

### 2.Load and Explore the Dataset
Load the California Housing Dataset using Scikit-Learn, perform exploratory data analysis (EDA) to understand data distributions, and check for correlations.

### 3.Data Preprocessing
Handle missing values (if any), and scale the features using StandardScaler for better model performance.

### 4.Train-Test Split
Split the dataset into training and testing sets using an 80-20 ratio.

### 5.Model Selection and Training
Train a regression model, such as Linear Regression or Random Forest Regressor, to predict housing prices based on various features.

### 6.Prediction and Evaluation
Evaluate the model using metrics such as Mean Squared Error (MSE) and R² score.

### 7.Cross-Validation
Use k-fold cross-validation to assess the model's generalizability.

### 8.Visualization of Results
Plot actual vs. predicted values to visually assess the model’s performance.

## Conclusion
This project demonstrates how to build a machine learning model to predict California housing prices. The model's performance can be further improved by experimenting with different algorithms and hyperparameters.
