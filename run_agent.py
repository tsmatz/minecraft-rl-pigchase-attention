import gym
import os
import pathlib
import random
import argparse
import time
import numpy as np

import ray
import ray.tune as tune
from ray.rllib import rollout
from ray.tune.registry import get_trainable_cls
from ray.rllib.models import ModelCatalog

from custom_model import MyCustomVisionNetwork
from custom_dist import MyCustomDiagGaussian
from custom_env import MyCustomGameEnv

# Generate mob's tag in mission xml
# ex.
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_dat',
        required=False,
        default=None,
        help="absolute directory path for world data",
        type=str)
    parser.add_argument('--checkpoint_file',
        required=False,
        default="./checkpoint/checkpoint-389",
        help="file path for trained checkpoint",
        type=str)
    parser.add_argument('--game_time_millisec',
        required=False,
        default=60000,
        help="game time milliseconds",
        type=int)
    parser.add_argument("--num_gpus",
        required=False,
        default=0,
        help="number of gpus",
        type=int)
    args = parser.parse_args()

    mission_file_path = str(pathlib.Path(__file__).parent.absolute()) + "/pigchase_mission.xml"
    if args.world_dat is None:
        args.world_dat = str(pathlib.Path(__file__).parent.absolute()) + "/flat_world"

    # Start with trained checkpoint
    ray.init()
    tune.register_env("testenv01", create_env)
    ModelCatalog.register_custom_action_dist("my_game_dist", MyCustomDiagGaussian)
    ModelCatalog.register_custom_model("my_game_model", MyCustomVisionNetwork)
    print("The world is starting ...")
    cls = get_trainable_cls("PPO")
    config={
        "env_config": {
            "mission_file": mission_file_path,
            "world_dat": args.world_dat,
            "shape": (120, 160, 3),
            "millisec_timeup": args.game_time_millisec,
            "millisec_per_tick": 50,
            "reward_damage": 80.0,
            "reward_action": -0.05,
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
    }
    agent = cls(env="testenv01", config=config)
    #agent.optimizer.stop()
    agent.restore(args.checkpoint_file)
    env = agent.workers.local_worker().env
    obs = env.reset()

    # Run agent
    done = False
    state=np.zeros((20, 64), np.float32)
    while not done:
        action, state, info = agent.compute_action(obs, [state], explore=True, full_fetch=True)
        # ## please uncomment when you monitor inputs for action dist
        # print(info["action_dist_inputs"])
        obs, _, done, _ = env.step(action)
    # Stop world
    input("Press Enter to stop world...")
    env.close()
    agent.stop()
    ray.shutdown()
    print("The world has stopped.")
