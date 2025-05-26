# generate_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create realistic sample data
np.random.seed(42)
n_samples = 1000

data = {
    'temperature': np.random.uniform(10, 35, n_samples),
    'humidity': np.random.uniform(40, 100, n_samples),
    'pressure': np.random.uniform(995, 1025, n_samples),
    'windspeed': np.random.uniform(0, 25, n_samples),
}

# Generate realistic rainfall relationships
data['rainfall'] = (
    0.8 * data['humidity'] + 
    0.5 * (100 - data['temperature']) + 
    0.3 * (1020 - data['pressure']) + 
    np.random.normal(0, 5, n_samples)
)
data['rainfall'] = np.clip(data['rainfall'], 0, 50)  # Keep between 0-50mm

# Convert to DataFrame
df = pd.DataFrame(data)

# Split features and target
X = df.drop('rainfall', axis=1)
y = df['rainfall']

# Create and train model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    ))
])

model.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rainfall_model.pkl')

print("Model successfully saved to models/rainfall_model.pkl")
print(f"Feature names: {model[:-1].get_feature_names_out()}")