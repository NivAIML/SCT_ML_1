# Import the required libraries
import pandas as pd  # To read CSV and work with data
import numpy as np
import matplotlib.pyplot as plt  # For visualizing
import seaborn as sns
from sklearn.model_selection import train_test_split  # To split data
from sklearn.linear_model import LinearRegression  # ML model
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation

# Load the dataset
data = pd.read_csv("house_data.csv")  # Your dataset file
print("First 5 rows of data:")
print(data.head())

# Check if any column has missing values
print("\nMissing values:")
print(data.isnull().sum())

# Extract input features (X) and target (y)
X = data[['square_feet', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Compare actual vs predicted
results = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})
print("\nComparison of actual vs predicted prices:")
print(results.head())

# Visualize
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
