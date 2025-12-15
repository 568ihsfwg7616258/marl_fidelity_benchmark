# src/policies
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

class TransparentMARLPolicy:       
    def __init__(self):
        self.tree = DecisionTreeClassifier(max_depth=8, random_state=42)
        print("Transparent Decision Tree Policy is ready!")

    def fit(self, observations, actions):
        self.tree.fit(observations, actions)
        print("Decision tree trained!")

    def predict(self, obs):
        return self.tree.predict([obs])[0]
    
    def save(self, path="models/transparent_dt.pkl"):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.tree, path)
        print(f"Transparent model saved at {path}")

    def get_ground_truth_attribution(self, obs):
        # This is exactly what Miró-Nicolau did!
        return self.tree.feature_importances_   # ← GROUND TRUTH attribution!