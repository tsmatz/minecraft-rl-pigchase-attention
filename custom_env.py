#################### ATTENTION
# The value reward_action is temporarily -0.1 in train.py
# (It usually is -0.05)
####################

import gym
import malmo.MalmoPython as MalmoPython
import random
import time
import json
import numpy as np
from enum import Enum
import ray
import math
import os
import sys
import pathlib
import subprocess
import signal
import socket
import traceback

CLIENT_PORT = 9000                  # Malmo port
TIME_WAIT = 0.05                    # Time to wait for retreiving world state (when MsPerTick=20)
MAX_LOOP = 50                       # Wait till TIME_WAIT * MAX_LOOP seconds for each action
FRAME_CHANNEL = 3                   # Channel size of converted observation frame

class MyCustomGameEnv(gym.Env):
    """
    A class implementing OpenAI gym environment
    to run both training and gaming in Agent Smith.
    This calls Project Malmo 0.36.0 Python API.

    init parameters
    ---------------
    xml : str (required)
        Mission setting (XML string) used in Project Malmo.
    world_dat : str (required)
        Absolute path for world data used in the game.
    shape : tuple (required)
        Shape for the frame returned in the game. e.g, (640, 480, 3)
    millisec_timeup : int (required)
        Milliseconds for mission time up.
    millisec_per_tick : int
        Milliseconds for each tick.
        In normal Minecraft game, it's 50 (default).
    mob_tag_func : function
        Function to generate mob's spawning tag for mission xml.
        (You can randomize in the training using this parameter.)
    reward_damage : float
        Positive reward for damaging mob.
    reward_action : float
        Negative reward for each action processing. (action penalty)
    reward_for_goal : float
        When the cumulative rewards reaches this value, the game will be ended.
    """
    def __init__(self,
        xml,
        world_dat,
        shape,
        millisec_timeup,
        millisec_per_tick=50,
        mob_tag_func=None,
        reward_damage=80.0,
        reward_action=-0.05,
        reward_for_goal=480.0):

        # Set up gym.Env
        super(MyCustomGameEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # turn degree [-1, 1]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(shape[0], shape[1], FRAME_CHANNEL), dtype=np.float64)
        # Initialize self variables
        self.xml = xml
        self.world_dat = world_dat
        self.shape = shape
        self.millisec_timeup = millisec_timeup
        self.millisec_per_tick = millisec_per_tick
        self.reward_damage = reward_damage
        self.reward_action = reward_action
        self.reward_for_goal = reward_for_goal
        self.mob_tag_func = mob_tag_func
        self.proc = None
        # Launch Minecraft client
        self._start_instance()
        # Create AgentHost
        self.agent_host = MalmoPython.AgentHost()
        # Create MissionRecordSpec
        self.my_mission_record = MalmoPython.MissionRecordSpec()
        self.my_mission_record.recordRewards()
        self.my_mission_record.recordObservations()
        # Create ClientPool
        self.pool = MalmoPython.ClientPool()
        client_info = MalmoPython.ClientInfo('127.0.0.1', CLIENT_PORT)
        self.pool.add(client_info)

    def __del__(self):
        self._kill_instance()

    """
    Public methods
    """

    def reset(self):
        # Create MissionSpec
        xml = self.xml
        xml = xml.format(self.world_dat, self.reward_damage, self.millisec_timeup, self.millisec_per_tick, self.mob_tag_func())
        my_mission = MalmoPython.MissionSpec(xml,True)
        # Start mission
        try:
            self.agent_host.startMission(my_mission,
                self.pool,
                self.my_mission_record,
                0,
                'test1')
        except:
            tb = traceback.format_exc()
            self._log("ERR : mission start err : {}".format(tb))
            self._kill_instance()
            time.sleep(3)
            self._start_instance()
            self.agent_host.startMission(my_mission,
                self.pool,
                self.my_mission_record,
                0,
                'test1')
        # Wait till mission begins
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(TIME_WAIT * self.millisec_per_tick / 20)
            world_state = self.agent_host.getWorldState()
        # Select slot 0 in hotbar inventory to take a diamond sword
        self.agent_host.sendCommand("hotbar.1 1")
        self.agent_host.sendCommand("hotbar.1 0")
        # Get reward, done, and frame
        frame, _, _ = self._process_state()
        if frame is None:
            obs = np.zeros(self.shape, dtype=np.uint8)
        else:
            obs = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(self.shape)
        # self.last_frame = self._get_frame_stack(obs)
        self.last_frame = obs / 255.0
        # Always moving !
        self.agent_host.sendCommand("forward 1")
        return self.last_frame

    def step(self, action):
        # Always attack !
        self.agent_host.sendCommand("attack 1")
        self.agent_host.sendCommand("attack 0")

        # Take action (turn)
        """
        turn degree is between -15 and 15, and positive number is right-turn.
        """
        if math.isnan(action):
            self._log("ERR : action is Nan")
            action = 0
        turn_value = int(action * 15.0 / 0.15)
        self.agent_host.sendCommand("moveMouse {} 0".format(turn_value))

        # Get reward, done, and frame
        frame, reward, done = self._process_state()
        # Penalty reward for each action (Used for negative gradient)
        reward = reward + self.reward_action
        # Clean up
        if done:
            frame3, reward3 = self._comsume_state()
            if frame3 is not None:
                frame = frame3
            if reward3 is not None:
                reward = reward + reward3
        # Generate frame
        if frame is None:
            obs = np.zeros(self.shape, dtype=np.uint8)
        else:
            obs = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(self.shape)
        self.last_frame = obs / 255.0

        return self.last_frame, reward, done, {}

    """
    Internal methods
    """

    # Extract frames, rewards, done_flag
    def _process_state(self):
        reward = 0
        frame_flag = False
        frame = None
        done = False
        loop = 0
        while True:
            # Just wait for one tick (The result might be delayed)
            time.sleep(0.001 * self.millisec_per_tick)
            # Get world state
            world_state = self.agent_host.getWorldState()
            # Get reward
            if world_state.number_of_rewards_since_last_state > 0:
                reward = reward + world_state.rewards[-1].getValue()
            # Get frame
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
                frame_flag = True
            # Get done flag
            done = not world_state.is_mission_running
            # Exit loop when extraction is completed
            if frame_flag:
                break;
            # Exit when MAX_LOOP exceeds
            # since done can be delayed...
            if done:
                loop = loop + 1
                if loop > MAX_LOOP:
                    break;
                time.sleep(TIME_WAIT * self.millisec_per_tick / 20)
        return frame, reward, done

    # Finalize episode (Clean-up)
    def _comsume_state(self):
        reward_flag = True
        reward = 0
        frame = None
        loop = 0
        while True:
            # Get next world state
            time.sleep(TIME_WAIT * self.millisec_per_tick / 10)
            world_state = self.agent_host.getWorldState()
            # Get reward (loop till command's rewards are all retrieved)
            if reward_flag and not (world_state.number_of_rewards_since_last_state > 0):
                reward_flag = False;
            if reward_flag:
                reward = reward + world_state.rewards[-1].getValue()
            # Get frame
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
            if not reward_flag:
                break;
        return frame, reward

    def _start_instance(self):
        # Check whether the port is already in use
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect(("127.0.0.1", CLIENT_PORT))
                s.close()
            print("Malmo port {} is already in use. Try to connect to existing Minecraft instance.".format(CLIENT_PORT))
            return
        except (ConnectionError, socket.timeout):
            print("Start Minecraft instance")

        # Launch Minecraft
        launch_shell_file = str(pathlib.Path(__file__).parent.absolute()) + "/launchClient.sh"
        dev_null = open(os.devnull, "w")
        self.proc = subprocess.Popen(
            ["bash", launch_shell_file, str(CLIENT_PORT)],
            stdout=dev_null,
            preexec_fn=os.setsid)

        # Wait till instance runs
        print("Waiting Minecraft instance to start ...")
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(10)
                    s.connect(("127.0.0.1", CLIENT_PORT))
                    s.close()
                print("Finished waiting for instance")
                break
            except (ConnectionError, socket.timeout):
                time.sleep(5)

    def _kill_instance(self):
        if self.proc is not None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc = None
            print("Terminated Minecraft instance")
