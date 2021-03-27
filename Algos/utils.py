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
    def __init__(self):
        self.policies = []
        self.qs = []
        self.ps = None
        self.maxQ = 1

        self.ELOS1 = []
        self.ELOS2 = []

        self.ELO1  = 1200
        self.ELO2  = 1200

        self.lr = 0.01

    def store_policy(self, policy):
        self.policies.append(policy)

        self.qs.append(self.maxQ)
        qs = np.array(self.qs)
        self.maxQ = np.max(qs)
        self.ps = np.exp(qs - self.maxQ) / np.sum(np.exp(qs - self.maxQ))

        self.ELOS1.append(self.ELO1)
        self.ELOS2.append(self.ELO2)

    def store_result(self, index, hasWon):
        SA = 1 if hasWon else 0
        SB = 0 if hasWon else 1

        self.ELOS1[index], self.ELO1 = self.updateELO(self.ELOS1[index], self.ELO1, SA, SB)
        _                , self.ELO2 = self.updateELO(self.ELOS2[index], self.ELO2, SA, SB)

        if hasWon: return

        self.qs[index] = self.qs[index] - self.lr / len(self) / self.ps[index]
        qs = np.array(self.qs)
        self.maxQ = np.max(qs)
        self.ps = np.exp(qs - self.maxQ) / np.sum(np.exp(qs - self.maxQ))
        
    def sample(self):
        assert len(self) > 0
        return np.random.choice(len(self), p=self.ps)
    
    def __len__(self):
        return len(self.policies)

    def updateELO(self, RA, RB, SA, SB):
        EA = 1 / (1 + 10 ** ((RB - RA) / 400))
        EB = 1 / (1 + 10 ** ((RA - RB) / 400))
        return RA + 32 * (SA - EA), RB + 32 * (SB - EB)

