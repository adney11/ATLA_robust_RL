from gym.envs.registration import register

register(
     id="PensieveGym-v0",
     entry_point="pensievegym.env_dir:PensieveGym",
     kwargs={},
     max_episode_steps=None,
)


