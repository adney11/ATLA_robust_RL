from real_systems.pensieve import PensieveEnv

from gym.envs.registration import register


register(
    id="PensieveEnv-v1",
    entry_point="real_systems:PensieveEnv"
)