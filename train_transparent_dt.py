# train_transparent_dt.py
import pickle
import numpy as np
from src.policies.transparent_dt import TransparentMARLPolicy
import os

# بارگذاری داده
with open("data/observations.pkl", "rb") as f:
    obs = pickle.load(f)
with open("data/actions.pkl", "rb") as f:
    actions = pickle.load(f)

# تبدیل به آرایه
X = np.array(obs)
y = np.array(actions)

# فقط نمونه‌هایی که action معتبر دارن (نه -1)
valid = y != -1
X = X[valid]
y = y[valid]

print(f"آموزش decision tree روی {len(X)} نمونه...")

policy = TransparentMARLPolicy()
policy.fit(X, y)
policy.save("models/transparent_dt.pkl")

print("ground-truth attribution آماده شد!")
print("اهمیت ویژگی‌ها (نمونه):")
print(policy.get_ground_truth_attribution(X[0])[:20])  # ۲۰ ویژگی اول