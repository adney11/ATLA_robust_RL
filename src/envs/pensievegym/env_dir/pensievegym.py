from .env import Environment, A_DIM, S_INFO, S_LEN, BITRATE_LEVELS

from .env import load_trace

import gym
from gym import spaces
import numpy as np
import os

import torch as ch


DEFAULT_QUALITY = 1  # default video quality without agent

VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0



class PensieveGym(gym.Env):
    def __init__(self):
        #all_cooked_time, all_cooked_bw, _ = load_trace()
        #assert len(all_cooked_time) == len(all_cooked_bw)
        super(PensieveGym, self).__init__()
        
        #self.action_space = spaces.Box(low=0, high=A_DIM-1, shape=(1,), dtype=np.int64)
        self.action_space = spaces.Discrete(A_DIM)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(S_INFO * S_LEN, ), dtype=np.float64)
        
        
        all_cooked_time, all_cooked_bw, _ = load_trace()
        
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        
        self.net_env = Environment(all_cooked_time = all_cooked_time, all_cooked_bw=all_cooked_bw)
        
        self.state = ch.zeros((S_INFO, S_LEN))
        
        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.last_bit_rate = DEFAULT_QUALITY
        
        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        
        self.step_num = 0
        
        self.video_size = self.net_env.video_size
        
    def _get_next_chunk_sizes(self):
        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
        return next_video_chunk_sizes
    
    def get_dims(self):
        return self.observation_space.shape
        
    def step(self, action):
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                self.net_env.get_video_chunk(action)
                
         # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[action] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[action] 
                    - VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K
        self.last_bit_rate = action
        
        self.state = ch.roll(self.state, -1, 1)
        
        
        self.state[0, -1] = VIDEO_BIT_RATE[action] / float(np.max(VIDEO_BIT_RATE))  # last quality
        self.state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        self.state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        self.state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec 
        self.state[4, :A_DIM] = ch.FloatTensor(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        self.state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        
        info = {
            "step_num": self.step_num,
            "action" : action,
            "reward" : reward,
            "buffer_size": buffer_size,
            "rebuf" : rebuf,
            "video_chunk_size" : video_chunk_size
        }
        
        self.step_num += 1
        
        return (self.state.flatten().numpy(), reward, end_of_video, info)

    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.video_chunk_counter = 0
        self.buffer_size = 0
        
        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the video
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = ch.zeros((S_INFO, S_LEN))
        
        self.step_num = 0
        info = {"msg": "Environment Reset"}
        return self.state.flatten().numpy()
        
        
        
    def render(self, mode="human", close=False):
        pass
        
        
if __name__ == "__main__":
    env = gym.make("")
