## Exercise 2 – Decision Trees

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier         # CART Clasifier
from sklearn import metrics
from sklearn.tree import plot_tree

# Load data
mech_data = pd.read_csv(r'/Users/lucasbroesamle/Documents/01_Studium/Master/02_Semester/Machine_Learning/Exercises/Source_Data/mechanical_component_failure_500.csv')

# Prepare data for modelling
X = mech_data[["Temperature", "Vibration", "Load"]]
y = mech_data["Failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Create Decision Tree classifer object
dtc = DecisionTreeClassifier(ccp_alpha=0.1)

# Train model
dtc.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = dtc.predict(X_test)

# Evaluate the model performance
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Visualize decision tree
plt.figure(figsize=(40,20))
plot_tree(dtc, 
          feature_names=X.columns, 
          class_names=["No Failure", "Failure"],
          filled=True,
          rounded=True)
plt.show()