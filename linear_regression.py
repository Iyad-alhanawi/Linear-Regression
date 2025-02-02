# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("Store_Profits.csv")

# Display information about the dataset
data.info()

# Display the dataset
data

# One-hot encode the 'State' column and drop the original column
data = data.join(pd.get_dummies(data.State)).drop(['State'], axis=1)

# Prepare the features and target variable
x = data.drop(['Profit'], axis=1)
y = data['Profit']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize and train the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions on the training and testing sets
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Plotting the predicted vs actual values for the training set
plt.figure(figsize=(5, 5))
plt.scatter(y_train, y_lr_train_pred)
plt.xlabel('Real Profit')
plt.ylabel('Predicted Profit')
plt.show()

# Calculate and display evaluation metrics
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Print the evaluation metrics
print('LR MSE (Train): ', lr_train_mse)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR R2 (Test): ', lr_test_r2)