from src.envs.wiod_env import WIODEnv_v2
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import numpy as np
from pathlib import Path
import json
import sys
import pathlib
import os
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# setting
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist


class ComprehensiveTrainingLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_sr = []
        self.episode_edges_removed = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_sr = 0
        self.current_episode_edges_removed = 0
        self.csv_path = RESULTS_DIR / "training_log.csv"  # CSV file path

    def _on_step(self) -> bool:
        # Collect metrics from the environment
        info = self.locals.get("infos", [{}])[0]
        reward = self.locals.get("rewards", [0])[0]

        self.current_episode_reward += reward
        self.current_episode_length += 1

       # Additional metrics from info
        sr = info.get("sr", 0)
        edges_removed = info.get("edges_removed", 0)
        self.current_episode_sr = sr
        self.current_episode_edges_removed = edges_removed

        # If the episode is over
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_sr.append(self.current_episode_sr)
            self.episode_edges_removed.append(
                self.current_episode_edges_removed)

            print(f" episode {len(self.episode_rewards)}: reward={self.current_episode_reward:.2f}, sr={self.current_episode_sr:.4f}, edges_removed={self.current_episode_edges_removed}")

            # Save metrics to CSV
            self._save_to_csv()

            # Reset variables
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_sr = 0
            self.current_episode_edges_removed = 0

        return True

    def _save_to_csv(self):
        import pandas as pd
        try:
            data = {
                "episode": range(len(self.episode_rewards)),
                "reward": self.episode_rewards,
                "length": self.episode_lengths,
                "sr": self.episode_sr,
                "edges_removed": self.episode_edges_removed
            }
            df = pd.DataFrame(data)
            print(f" Save training logs to {self.csv_path}...")
            print(f" Number of logged episodes: {len(self.episode_rewards)}")
            df.to_csv(self.csv_path, index=False)
            print(f" Logs saved successfully !")
        except Exception as e:
            print(f" Error saving CSV: {str(e)}")
            raise


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


def train():
    try:
        # Dynamic calculation of target_sr (from previous outputs)
        # target_sr = 0.50  # I got it from previous logs
        print(f"\n Start training with the following specifications:")
        # print(f"   target_sr: {env.target_sr:.4f}")
        print(f"   shock_mode: True")
        print(f"   max_steps: 20")
        print(f"   countries: ['CHN', 'USA', 'TWN', 'MEX']")
        print(f"   sectors: 56")

        print(" Building the environment WIODEnv_v2...")
        env = WIODEnv_v2(
            shock_mode=True,
            max_steps=20
        )
        obs, info = env.reset()
        print(" The environment was created successfully !")

        print(" Building a model PPO...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cpu",
            n_steps=2048,
            batch_size=128,
            learning_rate=3e-4,
            tensorboard_log=None,  # Disabled
            gamma=0.99,
            ent_coef=0.01
        )
        print(" PPO model successfully built !")

        logger_callback = ComprehensiveTrainingLogger()
        progress_callback = ProgressBarCallback(50_000)

        print(" Save training settings...")
        training_config = {
            "target_sr": env.target_sr,
            "shock_mode": True,
            "max_steps": 20,
            "countries": ['CHN', 'USA', 'TWN', 'MEX'],
            "sectors": 56,
            "n_steps": 2048,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "total_timesteps": 50_000
        }
        with open(RESULTS_DIR / "training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)
        print(" Settings saved !")

        print(" Training has begun....")
        model.learn(
            total_timesteps=50_000,
            callback=[logger_callback, progress_callback],
            tb_log_name=f"ppo_wiod_target_{env.target_sr:.4f}_shock"
        )

        print(" Save the trained model...")
        model.save(RESULTS_DIR / "ppo_wiod_model")
        print(" Model saved !")

    except Exception as e:
        print(f" Error in training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    train()
