# src/policies/ppo_blackbox.py
from pettingzoo.mpe import simple_spread_v3
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

class PPOBlackboxPolicy:
    def __init__(self):
        # ساخت محیط PettingZoo
        env = simple_spread_v3.parallel_env(N=3, max_cycles=100, continuous_actions=False)
        
        # Wrapperهای ضروری
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.agent_indicator_v0(env)
        env = ss.black_death_v3(env)
        
        # تبدیل به VecEnv برای SB3
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 4, num_cpus=0, base_class="stable_baselines3")  # num_cpus=0 برای ویندوز
        env = VecMonitor(env)
        
        self.model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
        print("PPO Black-box Policy با موفقیت ساخته شد!")

    def train(self, total_timesteps=20000):
        print("آموزش black-box شروع شد...")
        self.model.learn(total_timesteps=total_timesteps)
        print("آموزش تموم شد!")

    def save(self, path="models/ppo_blackbox"):
        self.model.save(path)
        print(f"مدل ذخیره شد: {path}")