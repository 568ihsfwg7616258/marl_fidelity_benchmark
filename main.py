# # main.py
# from src.policies.ppo_blackbox import PPOBlackboxPolicy
# from src.policies.transparent_dt import TransparentMARLPolicy

# if __name__ == "__main__":
#     print("=== تست black-box policy ===")
#     ppo = PPOBlackboxPolicy()
    
#     # فقط یه دور آموزش کوتاه برای تست
#     ppo.train(timesteps=8000)  # ۲ دقیقه طول می‌کشه
#     ppo.save()
    
#     print("=== همه چیز کار کرد! حالا بریم ground-truth بسازیم ===")

# main.py


# main.py


# main.py


from src.policies.ppo_blackbox import PPOBlackboxPolicy

if __name__ == "__main__":
    print("=== تست نهایی PPO Black-box در MARL ===")
    ppo = PPOBlackboxPolicy()
    ppo.train(total_timesteps=10000)  # فقط ۲–۳ دقیقه
    ppo.save()
    print("همه چیز کار کرد! حالا برو ground-truth بساز")