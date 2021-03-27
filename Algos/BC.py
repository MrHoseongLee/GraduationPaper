import gym
import torch as T
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def load_data_PPO(batch_size, device):
    obs = T.from_numpy(np.load('Data/PPO/obs.npy').astype('float32')).to(device)
    act = T.from_numpy(np.load('Data/PPO/act.npy').astype('long')).to(device)
    new = T.from_numpy(np.load('Data/PPO/new.npy').astype('float32')).to(device)
    rew = T.from_numpy(np.load('Data/PPO/rew.npy').astype('float32')).to(device)
    don = T.from_numpy(np.load('Data/PPO/don.npy').astype('bool')).to(device)
    dataset = TensorDataset(obs, act, new, rew, don)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_PPO(net, optim, device, gamma, lmbda, epochs, batch_size):
    train = load_data_PPO(batch_size, device)

    p_critirion = T.nn.CrossEntropyLoss()
    v_critirion = T.nn.SmoothL1Loss()

    for epoch in range(epochs):

        lossPTSum, lossVTSum, corrTSum = 0, 0, 0
        lossPVSum, lossVVSum, corrVSum = 0, 0, 0
        
        for obs, act, new, rew, don in train:

            old_p, old_v = net(obs)
            new_p, new_v = net(new)

            delta = (rew + gamma * new_v * (~don) - old_v).cpu().detach().numpy()

            advantage_list = []
            advantage = 0.0

            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])

            advantage_list.reverse()
            advantage = T.tensor(advantage_list, dtype=T.float32, device=device)

            tar_v = (advantage + old_v).detach()

            p_loss = p_critirion(old_p, act)
            v_loss = v_critirion(old_v, tar_v)

            loss = p_loss + v_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            lossPTSum += p_loss.detach().cpu().item()
            lossVTSum += v_loss.detach().cpu().item()

            corrTSum += np.sum(np.argmax(old_p.detach().cpu().numpy(), axis=1) == act.cpu().numpy())

        losPT = lossPTSum / len(train)
        losVT = lossVTSum / len(train)
        accT = corrTSum / len(train.dataset) * 100

        print(f'\rBehvior Cloning Epoch: {epoch+1:02}\t P Loss: {losPT:5f}' +\
                f'\t V Loss: {losVT:5f}\t Acc: {accT:5f}%', end='')
        print()
