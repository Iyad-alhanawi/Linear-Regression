# Linear-Regression


## Overview


- **Data Loading and Preprocessing:**
  - Loads a dataset named "Store_Profits.csv" using pandas to checks for null or damaged values in the dataset..
  - One-hot encodes the categorical feature "State" and drops the original column.

- **Data Splitting:**
  - Splits the dataset into features (x) and the target variable (y) for model training.
  - Splits the data into training and testing sets using the `train_test_split` function from scikit-learn.

- **Model Training:**
  - Initializes a Linear Regression model from scikit-learn.
  - Fits the model using the training data (x_train, y_train).

- **Model Evaluation:**
  - Makes predictions on the training and testing sets using the trained model.
  - Calculates Mean Squared Error (MSE) and R² (Coefficient of Determination) scores.

- **Visualization:**
  - Plots a scatter plot of real profit values vs. predicted profit values for the training set.

- **Results Display:**
  - Prints the MSE and R² scores for the Linear Regression model on both the training and testing sets.





