## Exercise 3 – Random Forest

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv(r'/Users/lucasbroesamle/Documents/01_Studium/Master/02_Semester/Machine_Learning/Exercises/Source_Data/automotive_dataset.csv')

## Baseline model: Decision Tree Regressor
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("price", axis=1), data["price"], test_size=0.2, random_state=42)

# Setting up the decision tree regressor
dt = DecisionTreeRegressor(random_state=42)

# Train the decision tree regressor
dt.fit(X_train, y_train)

# Predict the price values for the test set using decision tree
y_pred = dt.predict(X_test)

# Metrics for decision tree
dt_mse = mean_squared_error(y_test, y_pred)
print(f"BaseDecision Tree MSE: {dt_mse:.2f}")

# Setting up the random forest regressor with 100 trees
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42)
rf.fit(X_train, y_train)

# Predict the price values for the test set using random forest
rf_pred = rf.predict(X_test)

# Metrics for random forest
rf_mse = mean_squared_error(y_test, rf_pred)
print(f"Base Random Forest MSE: {rf_mse:.2f}")

## Tuning the models
# Tuning the decision tree with max_depth of 5
dt_tuned = DecisionTreeRegressor(random_state=42, max_leaf_nodes=10, ccp_alpha=0.001)
dt_tuned.fit(X_train, y_train)
dt_tuned_pred = dt_tuned.predict(X_test)
# Metrics for decision tree
dt_mse_tuned = mean_squared_error(y_test, dt_tuned_pred)
print(f"Tuned Decision Tree MSE: {dt_mse_tuned:.2f}")

# Random Forest with 500 trees and max_features of 0.3
rf_tuned = RandomForestRegressor(n_estimators=1000, max_features=7, random_state=42)
rf_tuned.fit(X_train, y_train)
rf_tuned_pred = rf_tuned.predict(X_test)
# Metrics for random forest
rf_mse_tuned = mean_squared_error(y_test, rf_tuned_pred)
print(f"Tuned Random Forest MSE: {rf_mse_tuned:.2f}")

# Feature importances
importances = rf.feature_importances_

for feature, importance in zip(X_train.columns, importances):
    print(f"{feature}: {importance}")

# Sort features
sorted_indices = importances.argsort()[::-1]
sorted_importances = importances[sorted_indices]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), sorted_importances, align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()