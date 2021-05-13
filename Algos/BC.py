import torch as T
import numpy as np
from torch.nn import NLLLoss 
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_data_PPO(batch_size, train_len, data_type, device):
    obs = T.from_numpy(np.load(f'Data/{data_type}/PPO/obs.npy').astype('float32')).to(device)
    act = T.from_numpy(np.load(f'Data/{data_type}/PPO/act.npy').astype('long')).to(device)
    train, val = random_split(TensorDataset(obs, act), [train_len, len(obs) - train_len])
    return DataLoader(train, batch_size=batch_size, shuffle=True),\
            DataLoader(val, batch_size=batch_size, shuffle=False)

def train_PPO(net, optim, epochs, batch_size, train_len, data_type, device):
    train, val = load_data_PPO(batch_size, train_len, data_type, device)

    critirion = NLLLoss()

    for epoch in range(epochs):

        lossTSum, corrTSum = 0, 0
        lossVSum, corrVSum = 0, 0
        
        for obs, act in train:

            p, _ = net(obs)
            loss = critirion(T.log(p), act)

            optim.zero_grad()
            loss.backward()
            optim.step()

            lossTSum += loss.detach().cpu().item()
            corrTSum += np.sum(np.argmax(p.detach().cpu().numpy(), axis=1) == act.cpu().numpy())

        for obs, act in val:

            with T.no_grad(): p, _ = net(obs)
            with T.no_grad(): loss = critirion(T.log(p), act)

            lossVSum += loss.detach().cpu().item()
            corrVSum += np.sum(np.argmax(p.detach().cpu().numpy(), axis=1) == act.cpu().numpy())

        losPT = lossTSum / len(train)
        losVT = lossVSum / len(val)
        accPT = corrTSum / len(train.dataset) * 100
        accVT = corrVSum / len(val.dataset) * 100

        print(f'\rBC Epoch: {epoch+1:02}\t'+\
                f'PT Loss: {losPT:5f}\t AccT: {accPT:5f}%\t'+\
                f'PV Loss: {losVT:5f}\t AccV: {accVT:5f}%', end='')
        print()

