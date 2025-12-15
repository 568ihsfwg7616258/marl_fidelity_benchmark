# train_transparent_dt.py
import pickle
import numpy as np
from src.policies.transparent_dt import TransparentMARLPolicy
import os

#load data
with open("data/observations.pkl", "rb") as f:
    obs = pickle.load(f)
with open("data/actions.pkl", "rb") as f:
    actions = pickle.load(f)

#transfer to numpy arrays
X = np.array(obs)
y = np.array(actions)

# only samples with valid actions (not -1)
valid = y != -1
X = X[valid]
y = y[valid]

print(f"Training decision tree on {len(X)} samples...")

policy = TransparentMARLPolicy()
policy.fit(X, y)
policy.save("models/transparent_dt.pkl")

print("Ground-truth attribution is ready!")
print("Feature importances (sample):")
print(policy.get_ground_truth_attribution(X[0])[:20])  # first 20 features