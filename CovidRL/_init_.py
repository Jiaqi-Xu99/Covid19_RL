from gym.envs.registration import register

register(
    id='CovidWorld-v0',
    entry_point='CovidRL.envs.CovidEnvironment:CovidEnv',
    max_episode_steps=1000,
)
