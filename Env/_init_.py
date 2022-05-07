from gym.envs.registration import register

register(
    id='env/CovidWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=300,
)