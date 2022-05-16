import gym

env = gym.make("CovidWorld-v0")
observation, info = env.reset(seed=42, return_info=False)
for _ in range(1000):
    action = 0
    observation, reward, done, info = env.step(action)

if done:
    observation, info = env.reset(return_info=False)

env.close()
