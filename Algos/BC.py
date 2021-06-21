import torch as T
import numpy as np
from torch.nn import NLLLoss 
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.functional import binary_cross_entropy

def load_data_P(batch_size, train_len, data_type, device):
    obs = T.from_numpy(np.load(f'Data/{data_type}/PPO/obs.npy').astype('float32')).to(device)
    act = T.from_numpy(np.load(f'Data/{data_type}/PPO/act.npy').astype('long')).to(device)
    train, val = random_split(TensorDataset(obs, act), [train_len, len(obs) - train_len])
    return DataLoader(train, batch_size=batch_size, shuffle=True),\
            DataLoader(val, batch_size=batch_size, shuffle=False)

def load_data_G(batch_size, train_len, data_type, device):
    demos = T.from_numpy(np.load(f'Data/{data_type}/GAIL/demos.npy').astype('float32')).to(device)
    train, val = random_split(demos, [train_len, len(demos) - train_len])
    return DataLoader(train, batch_size=batch_size, shuffle=True),\
            DataLoader(val, batch_size=batch_size, shuffle=False)

def train_P(net, optim, epochs, batch_size, train_len, data_type, device):
    train, val = load_data_P(batch_size, train_len, data_type, device)

    critirion = NLLLoss()

    for epoch in range(epochs):

        lossTSum, corrTSum = 0, 0
        lossVSum, corrVSum = 0, 0
        
        for obs, act in train:

            p, *_ = net(obs)
            loss = critirion(T.log(p), act)

            optim.zero_grad()
            loss.backward()
            optim.step()

            lossTSum += loss.detach().cpu().item()
            corrTSum += np.sum(np.argmax(p.detach().cpu().numpy(), axis=1) == act.cpu().numpy())

        for obs, act in val:

            with T.no_grad(): p, *_ = net(obs)
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

def train_G(discrim, optim, epochs, batch_size, train_len, data_type, device):
    train, val = load_data_G(batch_size, train_len, data_type, device)

    for epoch in range(epochs):

        lossTSum, corrTSum = 0, 0
        lossVSum, corrVSum = 0, 0
        
        for demo in train:

            obs = demo[:, :12]
            act = T.randint(0, 12, (len(obs), 1), dtype=T.float32, device=device)

            expert = discrim(demo)
            learner = discrim(T.cat([obs, act], dim=1))

            loss = binary_cross_entropy(learner, T.ones_like(learner, device=device)) +\
                    binary_cross_entropy(expert, T.zeros_like(expert, device=device))

            optim.zero_grad()
            loss.backward()
            optim.step()

            lossTSum += loss.detach().cpu().item()

        for demo in val:

            obs = demo[:, :12]
            act = T.randint(0, 12, (len(obs), 1), dtype=T.float32, device=device)

            with T.no_grad(): expert = discrim(demo)
            with T.no_grad(): learner= discrim(demo)

            with T.no_grad(): 
                loss = binary_cross_entropy(learner, T.ones_like(learner, device=device)) +\
                        binary_cross_entropy(expert, T.zeros_like(expert, device=device))

            lossVSum += loss.detach().cpu().item()

        losPT = lossTSum / len(train)
        losVT = lossVSum / len(val)

        print(f'\rBC Epoch: {epoch+1:02}\t'+\
                f'PT Loss: {losPT:5f}\tPV Loss: {losVT:5f}', end='')
        print()

