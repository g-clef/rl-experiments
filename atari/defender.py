from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env("Defender-v4", n_envs=4, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)
model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")