## Exercise 1.1 – Regression analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
df = pd.read_csv(r'/Users/lucasbroesamle/Documents/01_Studium/Master/02_Semester/Machine_Learning/Exercises/Source_Data/linear_regression.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[["size", "insulation", "temperature"]], 
                                                    df["energy"], test_size=0.3, random_state=42)

# Setting up the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Print the coefficients of the model
print("Coefficients: ", model.coef_)

# Print the intercept of the model
print("Intercept: ", model.intercept_)

# Predict the energy values for the test set
y_predict = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_predict)

plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted Energy")

# Ideal line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

# Textbox with coefficients, intercept and metrics
textstr = (
    f"Coefficients:\n"
    f"size: {model.coef_[0]:.2f}\n"
    f"insulation: {model.coef_[1]:.2f}\n"
    f"temperature: {model.coef_[2]:.2f}\n\n"
    f"Intercept: {model.intercept_:.2f}\n\n"
    f"MSE: {mse:.2f}\n"
    f"MAE: {mae:.2f}\n"
    f"R²: {r2:.2f}"
)

# Position text box inside plot
plt.gca().text(
    0.05, 0.95, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict()
)

plt.grid()
plt.show()