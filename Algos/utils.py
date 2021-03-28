import torch as T
import numpy as np

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

class PolicyBuffer:
    def __init__(self, N):
        self.N = N
        self.lr = 0.01

        self.policies = [None] * self.N

        self.qs = np.zeros(self.N, dtype=np.float32)
        self.ps = None

        self.maxQ = 1

        self.idx = 0

    def store_policy(self, policy):
        self.policies[self.idx % self.N] = policy
        self.qs[self.idx % self.N] = self.maxQ

        self.idx += 1

        self.calculate_ps()

    def store_result(self, index, hasWon):
        if hasWon: return

        self.qs[index] = self.qs[index] - self.lr / len(self) / self.ps[index]
        self.maxQ = np.max(self.qs[:len(self)])
        self.calculate_ps()

    def calculate_ps(self):
        enqs = np.exp(self.qs[:len(self)] - self.maxQ)
        self.ps = enqs / np.sum(enqs)
        
    def sample(self):
        assert len(self) > 0
        return np.random.choice(len(self), p=self.ps)
    
    def save(self, PATH):
        np.save(PATH, np.pad(self.ps, (0, self.N - len(self)), constant_values=0))
    
    def __len__(self):
        return min(self.idx, self.N)

