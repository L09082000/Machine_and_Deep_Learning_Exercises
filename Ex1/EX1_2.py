## Exercise 1.2 – Regression analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv(r'/Users/lucasbroesamle/Documents/01_Studium/Master/02_Semester/Machine_Learning/Exercises/Source_Data/poly_regression.csv')
# Clean column names
df.columns = df.columns.str.strip()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[["Distance (m)"]],
    df["Deflection (mm)"], 
    test_size=0.3, 
    random_state=42
)

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Prediction
y_pred = model.predict(X_test_poly)

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot (Polynomial Curve + Data Points)
plt.figure(figsize=(8,6))

# Original data
plt.scatter(df["Distance (m)"], df["Deflection (mm)"], label="Data")

# Smooth curve
X_plot = np.linspace(df["Distance (m)"].min(), df["Distance (m)"].max(), 100).reshape(-1,1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.plot(X_plot, y_plot, label="Polynomial fit")

plt.xlabel("Distance (m)")
plt.ylabel("Deflection (mm)")
plt.title("Polynomial Regression")

# Textbox
textstr = (
    f"Coefficients:\n"
    f"1: {model.coef_[0]:.2f}\n"
    f"x: {model.coef_[1]:.2f}\n"
    f"x²: {model.coef_[2]:.2f}\n\n"
    f"Intercept: {model.intercept_:.2f}\n\n"
    f"MSE: {mse:.2f}\n"
    f"MAE: {mae:.2f}\n"
    f"R²: {r2:.2f}"
)

plt.gca().text(
    0.2, 0.5, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict()
)

plt.legend()
plt.grid()
plt.show()