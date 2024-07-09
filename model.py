import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('data/us-population-2010-2019-reshaped.csv')
# Assuming 'df' is your DataFrame and it includes 'previous_year_population', 'economic_indicator', and 'population' columns
X = df[['previous_year_population', 'economic_indicator']]  # Predictor variables
y = df['population']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error: {rmse}")