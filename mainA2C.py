# Instantiate the env
from quoridor_env import QuoridorEnv
import os

# Define and Train the agent

from stable_baselines3 import A2C

env = QuoridorEnv()

algo = "A2C"

models_dir = "models/" + algo
logs_dir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

TIMESTEPS = 10000
i = 0
while True:
    i += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algo)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

'''
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
'''