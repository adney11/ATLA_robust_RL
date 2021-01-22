import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Hopper-v2"],
    "mode": ["ppo"],
    "out_dir": ["lstm_ppo_hopper/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [False],
    "ppo_lr_adam": [0.001],
    "val_lr": [0.01],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [20],
    "train_steps": [976],
    "robust_ppo_eps": [0.075], # used for attack
    "history_length": [100],
    "use_lstm_val": [True],
    "adv_clip_eps": [0.2],
    "adv_entropy_coeff": [0.0],
    "adv_ppo_lr_adam": [1e-3],
    "adv_val_lr": [1e-3],
}

generate_configs(BASE_CONFIG, PARAMS)
