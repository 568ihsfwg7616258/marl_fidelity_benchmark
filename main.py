
from src.policies.ppo_blackbox import PPOBlackboxPolicy

if __name__ == "__main__":
    print("=== Final test of PPO Black-box in MARL ===")
    ppo = PPOBlackboxPolicy()
    ppo.train(total_timesteps=10000)  
    ppo.save()
    print("Everything worked! Now go build the ground-truth")