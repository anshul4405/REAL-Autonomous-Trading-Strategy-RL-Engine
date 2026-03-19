import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

class RLAgent:
    """Advanced Wrapper class for the LSTM-PPO agent from Stable-Baselines3 Contrib."""
    
    def __init__(self, env):
        # Stable Baselines3 requires a vectorized environment
        self.env = DummyVecEnv([lambda: env])
        
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
        """Train the LSTM-PPO agent."""
        print(f"Training Advanced LSTM-PPO agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed.")
        
    def save_model(self, path="models/lstm_ppo_trading_agent"):
        """Save the trained Recurrent agent."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"LSTM Model saved to {path}.zip")
        
    def load_model(self, path="models/lstm_ppo_trading_agent"):
        """Load an existing Recurrent agent."""
        if os.path.exists(f"{path}.zip"):
            self.model = RecurrentPPO.load(path, env=self.env)
            print(f"LSTM Model loaded from {path}.zip")
        else:
            raise FileNotFoundError(f"No model found at {path}.zip")
            
    def predict(self, observation, lstm_states=None, episode_start=None):
        """
        Get an action from the recurrent agent. 
        LSTM networks strictly require the hidden state vectors from the preceding predict call.
        """
        action, new_lstm_states = self.model.predict(
            observation, 
            state=lstm_states, 
            episode_start=episode_start,
            deterministic=True
        )
        return action, new_lstm_states
