import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from app import get_data
import numpy as np
from scipy.optimize import minimize

# Load data
df = get_data('models-tone')

# NOTE: prepare data

# Replace model_type PPO = 0 and DQN = 1
df['model_type'] = df['model_type'].replace({'PPO': 0, 'DQN': 1})
# Drop unnecessary columns
df = df.drop(columns=['net_arch', 'net_arch_dqn', 'model_name', 'map_size', 'policy', 'folder_path_for_models'])

# Separate features and target variable
X = df.drop('model_performance', axis=1)
y = df['model_performance']

# Convert 'y' to numeric if needed
y = pd.to_numeric(y, errors='coerce')  # Convert to numeric, handle errors by setting invalid parsing to NaN
y = y.dropna()  # Drop NaN values if any

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the distribution of the target variable
sns.histplot(y_train, bins=20, kde=True)
plt.title("Distribution of the Target Variable")
plt.xlabel("Target Variable")
plt.ylabel("Frequency")
# plt.show()

print('X_train:', X_train)
print('X_train:', X_train.shape)

# Normalize the target variable
# scaler = StandardScaler()
# y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

# joblib.dump(scaler, 'scaler.pkl')

# Define the parameter grid with correct 'criterion' values
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'criterion': ['friedman_mse', 'squared_error', 'poisson', 'absolute_error']
}

# Train the model with GridSearchCV
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# evaluate the model
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
accuracy = grid_search.score(X_test, y_test)
print("Accuracy:", accuracy)

def objective(params, model):
    params_reshaped = np.array(params)
    # reshape to 2D array
    params_reshaped = params_reshaped.reshape(1, -1)

    # Predict the performance using the model
    predicted_performance = model.predict(params_reshaped)

    # Return the negative performance (since we are minimizing)
    return -predicted_performance[0]

# loaded_scaler = joblib.load('scaler.pkl')

# Use the best-trained model from the grid search
best_model = grid_search.best_estimator_

# Define initial parameters as an array of zeros
initial_params = np.zeros(X_train.shape[1])

# Perform the minimization to find the best parameters
result = minimize(objective, initial_params, args=(best_model), method='Powell')

# Get the best parameters
best_params = result.x

# Scale the best parameters using the loaded scaler
best_params_reshaped = np.array(best_params).reshape(1, -1)
# best_params_scaled = loaded_scaler.transform(best_params_reshaped)

# Predict the performance using the best model
predicted_performance = best_model.predict(best_params_reshaped)

new_df = pd.DataFrame(best_params_reshaped, columns=X_train.columns)
print("Best Parameters:", new_df)

print("Predicted Performance:", predicted_performance[0])

