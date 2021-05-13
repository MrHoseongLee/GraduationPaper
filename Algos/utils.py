import torch as T
import numpy as np
from collections import deque

class ReplayBufferPPO():
    def __init__(self, buffer_size, obs_dim, device):
        self.device = device

        self.obs_arr = T.zeros((buffer_size, obs_dim), dtype=T.float32, device=device)
        self.new_arr = T.zeros((buffer_size, obs_dim), dtype=T.float32, device=device)
        self.act_arr = T.zeros((buffer_size, 1), dtype=T.int64, device=device)
        self.rew_arr = T.zeros((buffer_size, 1), dtype=T.float32, device=device)
        self.prb_arr = T.zeros((buffer_size, 1), dtype=T.float32, device=device)
        self.don_arr = T.zeros((buffer_size, 1), dtype=T.bool, device=device)

        self.mem_cnt = 0
        self.buffer_size = buffer_size

    def store_data(self, obs, act, rew, new, prb, don):
        self.obs_arr[self.mem_cnt]    = obs
        self.new_arr[self.mem_cnt]    = new
        self.act_arr[self.mem_cnt][0] = act
        self.rew_arr[self.mem_cnt][0] = rew
        self.prb_arr[self.mem_cnt][0] = prb
        self.don_arr[self.mem_cnt][0] = don

        self.mem_cnt += 1

    @property
    def isFull(self):
        return self.mem_cnt == self.buffer_size

    def sample(self):
        return self.obs_arr, self.act_arr, self.new_arr, self.rew_arr, self.prb_arr, self.don_arr

    def reset(self):
        self.mem_cnt = 0

        self.obs_arr.zero_()
        self.new_arr.zero_()
        self.act_arr.zero_()
        self.rew_arr.zero_()
        self.prb_arr.zero_()
        self.don_arr.zero_()

class ReplayBufferTHGAIL():
    def __init__(self, buffer_size, obs_dim, device):
        self.device = device

        self.obs_arr = T.zeros((buffer_size, obs_dim), dtype=T.float32, device=device)
        self.new_arr = T.zeros((buffer_size, obs_dim), dtype=T.float32, device=device)
        self.act_arr = T.zeros((buffer_size, 1), dtype=T.int64, device=device)
        self.irl_arr = T.zeros((buffer_size, 1), dtype=T.float32, device=device)
        self.rew_arr = T.zeros((buffer_size, 1), dtype=T.float32, device=device)
        self.prb_arr = T.zeros((buffer_size, 1), dtype=T.float32, device=device)
        self.don_arr = T.zeros((buffer_size, 1), dtype=T.bool, device=device)

        self.mem_cnt = 0
        self.buffer_size = buffer_size

    def store_data(self, obs, act, irl, rew, new, prb, don):
        self.obs_arr[self.mem_cnt]    = obs
        self.new_arr[self.mem_cnt]    = new
        self.act_arr[self.mem_cnt][0] = act
        self.irl_arr[self.mem_cnt][0] = irl
        self.rew_arr[self.mem_cnt][0] = rew
        self.prb_arr[self.mem_cnt][0] = prb
        self.don_arr[self.mem_cnt][0] = don

        self.mem_cnt += 1

    @property
    def isFull(self):
        return self.mem_cnt == self.buffer_size

    def sample(self):
        return self.obs_arr, self.act_arr, self.new_arr, self.irl_arr, self.rew_arr, self.prb_arr, self.don_arr

    def reset(self):
        self.mem_cnt = 0

        self.obs_arr.zero_()
        self.new_arr.zero_()
        self.act_arr.zero_()
        self.irl_arr.zero_()
        self.rew_arr.zero_()
        self.prb_arr.zero_()
        self.don_arr.zero_()

class PolicyBuffer:
    def __init__(self, N):
        self.N = N

        self.policies = deque(maxlen=self.N)

        self.ps = []

    def store_policy(self, policy):
        self.policies.append(policy)
        self.calculate_ps()

    def calculate_ps(self):
        if len(self.ps) == self.N: return
        qs, N = [], len(self)
        for i in range(N): qs.append(1 / (N * (N - i) ** 2))
        self.ps = np.array(qs) / sum(qs)

    def sample(self):
        assert len(self) > 0
        return np.random.choice(list(self.policies), p=self.ps)
    
    def __len__(self):
        return len(self.policies)

