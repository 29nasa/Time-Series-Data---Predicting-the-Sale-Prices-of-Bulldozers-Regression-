# Predicting Bulldozer Sale Prices using Machine Learning

## Project Overview
This project is a time series machine learning regression task focused on predicting the sale prices of bulldozers. The project is based on data from the Kaggle Bluebook for Bulldozers competition. The goal is to predict future auction prices for bulldozers using historical sales data and various machine learning models.

## Problem Definition
> How accurately can we predict the sale prices of bulldozers based on past sales and their respective features?

## Dataset
The dataset is sourced from the Kaggle Bluebook for Bulldozers competition and includes the following files:
- **Train.csv**: Contains sales data through the end of 2011.
- **Valid.csv**: Contains data from January 1, 2012, to April 30, 2012, used for validation.
- **Test.csv**: Contains data from May 1, 2012, to November 2012, used for testing.

You can download the dataset from Kaggle [here](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data).

## Evaluation Metric
The competition uses the **Root Mean Squared Logarithmic Error (RMSLE)** as the evaluation metric to assess the performance of the predictions.

## Key Features
- **SalesID**: Unique identifier for the sale.
- **SalePrice**: The target variable, representing the sale price of the bulldozer.
- **MachineID, ModelID, YearMade**: Information about the machine's specifications.
- **saledate**: The sale date of the bulldozer, which is crucial for time series analysis.
- **Other machine features**: Various categorical and numerical attributes related to bulldozers.

## Workflow Steps
1. **Data Loading & Cleaning**:
   - Handled missing values and categorical data by converting string data into numerical categories.
   - Managed time series data by parsing and extracting components like year, month, and day.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized key attributes like sales price distribution and sale date vs. sale price.
   - Checked for missing data and made necessary imputations.

3. **Feature Engineering**:
   - Added time-related features like sale year, sale month, and sale day.
   - Converted categorical variables into numeric categories to make them compatible with machine learning models.

4. **Modeling**:
   - Trained a **Random Forest Regressor** to predict sale prices.
   - Implemented **train/validation split** based on the year to avoid data leakage.
   - Utilized **GridSearchCV** to tune hyperparameters.

5. **Model Evaluation**:
   - Evaluated models using **MAE**, **RMSLE**, and **R2** scores to assess performance on the training and validation sets.

6. **Performance Improvement**:
   - Investigated feature importance and improved model efficiency by reducing computation time and using parallelization (`n_jobs=-1`).

## Results
- **Best RMSLE Score**: The Random Forest Regressor achieved a strong score of 0.225 on the validation set.
- **Top Features**: Features like "YearMade" and "saledate" had a significant impact on sale price predictions.

## Tools and Libraries
- **Python**: `pandas`, `NumPy`, `matplotlib`, `scikit-learn`
- **Machine Learning**: Random Forest Regressor
- **Hyperparameter Tuning**: `GridSearchCV`
- **Data Visualization**: `matplotlib`, `seaborn`

## How to Run This Project
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
