from gym.envs.registration import register

register(
    id='CovidWorld-v0',
    entry_point='env:CovidWorld',
    max_episode_steps=1000,
)