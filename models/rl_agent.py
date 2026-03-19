import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class RLAgent:
    """Wrapper class for the PPO agent from Stable-Baselines3"""
    def __init__(self, env):
        # Stable Baselines3 requires a vectorized environment
        self.env = DummyVecEnv([lambda: env])
        
        # Initialize PPO model with MlpPolicy for structured tabular data
        self.model = PPO(
            "MlpPolicy", 
            self.env, 
            verbose=1, 
            tensorboard_log="./tensorboard_logs/",
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99
        )
        
    def train(self, total_timesteps=10000):
        """Train the agent in the environment."""
        print(f"Training PPO agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed.")
        
    def save_model(self, path="models/ppo_trading_agent"):
        """Save the trained agent."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}.zip")
        
    def load_model(self, path="models/ppo_trading_agent"):
        """Load an existing agent from disk."""
        if os.path.exists(f"{path}.zip"):
            self.model = PPO.load(path, env=self.env)
            print(f"Model loaded from {path}.zip")
        else:
            raise FileNotFoundError(f"No model found at {path}.zip")
            
    def predict(self, observation):
        """Get an action from the agent given an observation."""
        action, _states = self.model.predict(observation, deterministic=True)
        return action
