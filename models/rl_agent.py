import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class RLAgent:
    """Advanced Wrapper class for the LSTM-PPO agent from Stable-Baselines3 Contrib."""
    
    def __init__(self, env):
        # Neural Networks struggle with raw pricing data ($1500 vs $0.0001). 
        # VecNormalize automatically scales observations (features) and rewards to a standard normal distribution.
        # This guarantees SUBSTANTIALLY better and faster convergence/predictions for the LSTM.
        self.raw_env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(self.raw_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Initialize RecurrentPPO model with MlpLstmPolicy for sequence memory over time
        self.model = RecurrentPPO(
            "MlpLstmPolicy", 
            self.env, 
            verbose=1, 
            tensorboard_log="./tensorboard_logs_lstm/",
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=64,
            gamma=0.99
        )
        
    def train(self, total_timesteps=15000):
        print(f"Training Advanced LSTM-PPO agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed.")
        
    def save_model(self, path="models/lstm_ppo_trading_agent"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        # Also save the normalizer statistics so live predictions map identically to training
        self.env.save(f"{path}_vecnormalize.pkl")
        print(f"LSTM Model & Normalizer saved to {path}")
        
    def predict(self, observation, lstm_states=None, episode_start=None):
        import numpy as np
        is_1d = len(np.shape(observation)) == 1
        if is_1d:
            observation = np.array([observation])
            
        norm_obs = self.env.normalize_obs(observation)
        action, new_lstm_states = self.model.predict(
            norm_obs, 
            state=lstm_states, 
            episode_start=episode_start,
            deterministic=True
        )
        
        if is_1d and isinstance(action, np.ndarray) and len(action) > 0:
            action = action[0]
            
        return action, new_lstm_states
