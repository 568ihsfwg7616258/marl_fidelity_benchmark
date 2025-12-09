# evaluate_fidelity.py
import pickle
import numpy as np
from src.evaluation.fidelity_metrics import measure_fidelity
from src.policies.transparent_dt import TransparentMARLPolicy
from stable_baselines3 import PPO
import joblib

# 1. بارگذاری مدل شفاف
gt_policy = TransparentMARLPolicy()
gt_policy.tree = joblib.load("models/transparent_dt.pkl")

# 2. بارگذاری black-box
blackbox_model = PPO.load("models/ppo_blackbox.zip")

# 3. یه نمونه تصادفی
with open("data/observations.pkl", "rb") as f:
    all_obs = pickle.load(f)
sample_obs = all_obs[100]

# 4. Ground-truth attribution
gt_attr = gt_policy.get_ground_truth_attribution(sample_obs)

# 5. Integrated Gradients (ساده)
# چون مدل PPO مستقیماً predict داره، از خروجی action استفاده می‌کنیم
action, _ = blackbox_model.predict(sample_obs, deterministic=True)
# برای IG نیاز به gradient داریم، پس از policy شبکه استفاده می‌کنیم
policy_net = blackbox_model.policy
obs_tensor = policy_net.obs_to_tensor([sample_obs])[0]

# یه attribution ساده (گرادیان نسبت به ورودی)
with torch.enable_grad():
    obs_tensor.requires_grad_()
    q_values = policy_net.get_distribution(obs_tensor).distribution.logits
    q_values.mean().backward()
    ig_attr = obs_tensor.grad.numpy().flatten()

# 6. محاسبه وفاداری
result = measure_fidelity(gt_attr, ig_attr)

print("="*60)
print("اولین جدول FIDELITY در MARL")
print("="*60)
for k, v in result.items():
    print(f"{k:25}: {v:.4f}")
print("="*60)
print("کار تموم شد! حالا می‌تونی به استاد نشون بدی")