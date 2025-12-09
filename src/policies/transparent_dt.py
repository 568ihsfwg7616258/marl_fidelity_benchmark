# src/policies/transparent_dt.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

class TransparentMARLPolicy:        # ← این اسم باید دقیقاً همین باشه!
    def __init__(self):
        self.tree = DecisionTreeClassifier(max_depth=8, random_state=42)
        print("Transparent Decision Tree Policy آماده شد!")

    def fit(self, observations, actions):
        self.tree.fit(observations, actions)
        print("درخت تصمیم آموزش دید!")

    def predict(self, obs):
        return self.tree.predict([obs])[0]
    
    def save(self, path="models/transparent_dt.pkl"):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.tree, path)
        print(f"مدل شفاف ذخیره شد: {path}")

    def get_ground_truth_attribution(self, obs):
        # این دقیقاً همون چیزیه که Miró-Nicolau کرد!
        return self.tree.feature_importances_   # ← GROUND TRUTH attribution!