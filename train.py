import gym
import numpy as np
import random
import pathlib
import argparse
import os

import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog

from custom_model import MyCustomVisionNetwork
from custom_dist import MyCustomDiagGaussian
from custom_env import MyCustomGameEnv

# Generate mob's tag in mission xml
# e.g,
#   <DrawEntity x="..." y="..." z="..." type="Pig"/>
#   <DrawEntity x="..." y="..." z="..." type="Pig"/>
#   ...
def _get_pig_tag():
    _MOB_NUM = 125

    mob_tag = ""
    for i in range(_MOB_NUM):
        # <DrawEntity x="{0[0]}" y="{0[1]}" z="{0[2]}" type="Pig"/>
        x = i * 3
        y = x + 1
        z = y + 1
        mob_tag += "<DrawEntity x=\"{0[" + str(x) + \
            "]}\" y=\"{0[" + str(y) + \
            "]}\" z=\"{0[" + str(z) + "]}\" type=\"Pig\"/>\n"

    mob_pos = np.empty((0,3), int)
    for i in range(_MOB_NUM):
        mob_pos = np.append(mob_pos, [[random.randint(662, 1174), 4, random.randint(-266, 246)]], axis=0)
    mob_pos = mob_pos.flatten()

    return mob_tag.format(mob_pos)

# For creating OpenAI Gym environment (MyCustomGameEnv)
def create_env(config):
    mission_file = config["mission_file"]
    xml = pathlib.Path(mission_file).read_text()
    world_dat = config["world_dat"]
    millisec_timeup = config["millisec_timeup"]
    millisec_per_tick = config["millisec_per_tick"]
    shape = config["shape"]
    reward_damage = config["reward_damage"]
    reward_action = config["reward_action"]
    env = MyCustomGameEnv(
        xml=xml,
        world_dat=world_dat,
        shape=shape,
        millisec_timeup=millisec_timeup,
        millisec_per_tick=millisec_per_tick,
        mob_tag_func=_get_pig_tag,
        reward_damage=reward_damage,
        reward_action=reward_action)
    return env

# For stopping a learner for successful training
def stop_check(trial_id, result):
    return result["episode_reward_mean"] >= 2000.0

@ray.remote
class EpisodeCounter:
   def __init__(self):
      self.count = 0
   def inc(self, n):
      self.count += n
   def get(self):
      return self.count

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus",
        type=int,
        required=False,
        default=0,
        help="number of gpus")
    args = parser.parse_args()

    mission_file_path = str(pathlib.Path(__file__).parent.absolute()) + "/pigchase_mission.xml"
    world_data_path = str(pathlib.Path(__file__).parent.absolute()) + "/flat_world"

    tune.register_env("testenv01", create_env)

    ray.init()

    ModelCatalog.register_custom_model("my_game_model", MyCustomVisionNetwork)
    ModelCatalog.register_custom_action_dist("my_game_dist", MyCustomDiagGaussian)

    tune.run(
        run_or_experiment="PPO",
        config={
            "log_level": "WARN",
            "env": "testenv01",
            "env_config": {
                "mission_file": mission_file_path,
                "world_dat": world_data_path,
                "shape": (120, 160, 3),
                "millisec_timeup": 60000,
                "millisec_per_tick": 20,
                "reward_damage": 80.0,
                "reward_action": -0.10,
            },
            "model": {
                "custom_model": "my_game_model",
                "custom_action_dist": "my_game_dist",
                "use_attention": True,
                "attention_num_transformer_units": 1,
                "attention_dim": 64,
                "attention_num_heads": 2,
                "attention_memory_inference": 100,
                "attention_memory_training": 50,
                "attention_use_n_prev_actions": 0,
                "attention_use_n_prev_rewards": 0,
            },
            "framework": "tf",
            "num_gpus": args.num_gpus,
            "num_workers": 0,
            "ignore_worker_failures": True,
            "train_batch_size": 4600,
            "vf_loss_coeff": 0.001,
            "explore": True,
        },
        stop=stop_check,
        checkpoint_freq=2,
        checkpoint_at_end=True,
        local_dir='./logs'
    )

    print('training has done !')
    ray.shutdown()
