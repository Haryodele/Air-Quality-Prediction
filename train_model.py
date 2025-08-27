import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
data = pd.read_csv("Air quality datasets.csv")

# Adjust column names based on your dataset
X = data[['temperature', 'humidity', 'pressure', 'wind']]
y = data['PM2.5_category']   # <- make sure this is the correct target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
