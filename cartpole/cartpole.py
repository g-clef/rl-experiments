import gymnasium as gym

from stable_baselines3 import A2C

# Reinforcement learning uses a state, a function to score the present state, and a
# function to take a step. The "gymnasium" offers a bunch of pre-defined problems,
# like cart-pole, atari games, lunar lander, etc. it also allows you to save and
# load state.
env = gym.make("CartPole-v1", render_mode="rgb_array")

# the model takes the gymnasium defined problem and applies a particular RL algorithm
# to it.
model = A2C("MlpPolicy", env, verbose=1)
# this is the learning stage, where it tries to figure out how to get better at the
# task at hand.
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")