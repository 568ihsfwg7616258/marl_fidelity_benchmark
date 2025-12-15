# evaluate_fidelity.py
import pickle
import numpy as np
import torch   
from src.evaluation.fidelity_metrics import measure_fidelity
from src.policies.transparent_dt import TransparentMARLPolicy
from stable_baselines3 import PPO
import joblib
import gymnasium as gym

# 1. Loaded (ground-truth)
gt_policy = TransparentMARLPolicy()
gt_policy.tree = joblib.load("models/transparent_dt.pkl")

# 2. loaded black-box
blackbox_model = PPO.load("models/ppo_blackbox.zip")

# 3. A random sample of data
with open("data/observations.pkl", "rb") as f:
    all_obs = pickle.load(f)
sample_obs = all_obs[100]

# 4. Ground-truth attribution
gt_attr = gt_policy.get_ground_truth_attribution(sample_obs)

# 5. Integrated Gradients (simple)
action, _ = blackbox_model.predict(sample_obs, deterministic=True)
policy_net = blackbox_model.policy
obs_tensor = policy_net.obs_to_tensor([sample_obs])[0]

with torch.enable_grad():
    obs_tensor.requires_grad_()
    q_values = policy_net.get_distribution(obs_tensor).distribution.logits
    q_values.mean().backward()
    ig_attr = obs_tensor.grad.numpy().flatten()

#measure_fidelity
result = measure_fidelity(gt_attr, ig_attr)

print("="*60)
print("First FIDELITY table in MARL")
print("="*60)
for k, v in result.items():
    print(f"{k:25}: {v:.4f}")
print("="*60)
print("Done! Now you can show it to the professor")
