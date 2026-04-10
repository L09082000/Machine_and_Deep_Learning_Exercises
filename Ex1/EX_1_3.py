## Exercise 1.2 – Regression analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv(r'/Users/lucasbroesamle/Documents/01_Studium/Master/02_Semester/Machine_Learning/Exercises/Source_Data/log_regression.csv')
# Clean column names
df.columns = df.columns.str.strip()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[["CO2_level", "NOx_level"]],
    df["Pass_emissions_test"], 
    test_size=0.9, 
    random_state=42
)

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_poly, y_train)

# Prediction
y_pred = model.predict(X_test_poly)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot
plt.figure(figsize=(8,6))

# Scatter data
plt.scatter(X_test["CO2_level"], X_test["NOx_level"], c=y_test, edgecolors='k')
plt.xlabel("CO2 level")
plt.ylabel("NOx level")
plt.title("Logistic Regression Decision Boundary")

# Meshgrid
x_min, x_max = X_test["CO2_level"].min() - 1, X_test["CO2_level"].max() + 1
y_min, y_max = X_test["NOx_level"].min() - 1, X_test["NOx_level"].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_poly = poly.transform(grid)

Z = model.predict(grid_poly)
Z = Z.reshape(xx.shape)

# Decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Textbox
textstr = (
    f"Accuracy: {accuracy:.2f}\n\n"
    f"Confusion Matrix:\n{conf_matrix}"
)

plt.gca().text(
    0.02, 0.98, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict()
)

plt.grid()
plt.show()