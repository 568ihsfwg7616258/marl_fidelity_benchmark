# collect_data.py
from pettingzoo.mpe import simple_spread_v3
import numpy as np
import pickle
import os

def collect_data(episodes=800):
    env = simple_spread_v3.env(N=3, max_cycles=100, continuous_actions=False)
    env.reset()
    
    observations = []
    actions = []
    
    print("جمع‌آوری داده شروع شد...")
    for ep in range(episodes):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample()
            
            observations.append(obs)
            actions.append(action if action is not None else -1)
            
            env.step(action)
        
        if (ep+1) % 100 == 0:
            print(f"اپیزود {ep+1}/{episodes} تموم شد")
    
    # ذخیره داده
    os.makedirs("data", exist_ok=True)
    with open("data/observations.pkl", "wb") as f:
        pickle.dump(observations, f)
    with open("data/actions.pkl", "wb") as f:
        pickle.dump(actions, f)
    
    print(f"داده جمع شد! تعداد نمونه: {len(observations)}")

if __name__ == "__main__":
    collect_data(episodes=800)  # حدود ۸–۱۲ دقیقه طول می‌کشه