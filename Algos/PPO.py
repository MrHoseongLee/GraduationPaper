# Imports
import torch as T
from torch.optim import Adam
from torch.distributions import Categorical
from torch.nn.functional import smooth_l1_loss

import json
import numpy as np

from Algos.Model.PPO import Model
from Algos.utils import ReplayBufferPPO, PolicyBuffer

def train(env, use_builtin, use_cuda, save_path):

    # Loading Config
    with open('Configs/config-PPO.json') as f: config = json.load(f)

    # PPO Config
    lr             = config['learning rate']      # Learing rate of the Actor/Critic
    steps          = config['total timestep']     # Total number of timesteps to train
    lmbda          = config['lambda']             # Lambda for GAE
    gamma          = config['gamma']              # Discount Factor
    epochs         = config['epochs']             # Num epochs for ppo update
    horizon        = config['horizon']            # Num steps before ppo update
    cliprange      = config['cliprange']          # Cliprange -> [0, 1]
    reward_coeff   = config['reward coeff']       # Reward Multiplier
    entropy_coeff  = config['entropy coeff']      # Entropy Coefficient

    # Self-Play Config
    P_capacity     = config['policy capacity']    # Size of policy buffer
    P_interval     = config['policy interval']    # Number of updates before storing new policy to buffer

    # BC Config
    use_BC         = config['use BC']             # Pretrain using BC or not
    BC_epochs      = config['BC epochs']          # Num epochs to run BC
    BC_batch_size  = config['BC batch size']      # Batch size used in BC
    BC_train_len   = config['BC train len']       # How much training data to use (rest is validation)
    BC_data_type   = config['BC data type']       # What data to use for training (Human or AI)

    # Save Config
    save_interval  = config['save interval']      # Num steps before saving

    # Set device used for training
    device = T.device('cuda') if use_cuda else T.device('cpu')

    # Create the learning and opposing agents
    opp = Model()
    net = Model()

    opp.to(device)
    net.to(device)

    optim = Adam(net.parameters(), lr=lr)
    
    # Pretrain using Behavior Cloning
    if use_BC: 
        from Algos.BC import train_PPO
        train_PPO(net, optim, BC_epochs, BC_batch_size, BC_train_len, BC_data_type, device)

    # Set starting opponent equal to agent
    if use_builtin: del opp
    else: opp.load_state_dict(net.state_dict())

    # Final touches before self-play starts
    if not use_builtin:
        policy_buffer = PolicyBuffer(P_capacity)
        policy_buffer.store_policy(net.state_dict())

    replay_buffer = ReplayBufferPPO(buffer_size=horizon, obs_dim=12, device=device)

    game_start = 0
    isPlayer2Serve = False

    observation = T.tensor(env.reset(not isPlayer2Serve), dtype=T.float32, device=device)

    a1 = T.tensor(0, dtype=T.int64, device=device)

    # Main Loop
    for step in range(steps):

        with T.no_grad():
            if not use_builtin: p1, *_ = opp(observation[0])
            p2, v2 = net(observation[1])

        if not use_builtin: a1 = Categorical(p1).sample()
        a2 = Categorical(p2).sample()

        action = T.stack((a1, a2), dim=-1).cpu().detach().numpy()

        new_observation, reward, done, _ = env.step(action)

        new_observation = T.tensor(new_observation, dtype=T.float32, device=device)
        reward = T.tensor(reward_coeff * reward, dtype=T.float32, device=device)
        done = T.tensor(done, dtype=T.bool, device=device)

        replay_buffer.store_data(observation[1], a2, reward, new_observation[1], p2[a2], done)

        observation = new_observation

        if done:
            observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32, device=device)

            isPlayer2Serve = not isPlayer2Serve

            print('\r', end=' ' * 80)
            print(f'\rCurrent Frame: {step+1}\tGame Length: {step-game_start}', end='')

            game_start = step

            # Opponent Sampling (delta-Limit Uniform)
            if isPlayer2Serve and not use_builtin:
                opp.load_state_dict(policy_buffer.sample())

        if not replay_buffer.isFull:
            continue
        
        # Sample Data
        obs, act, new, rew, prb, don = replay_buffer.sample()

        # PPO Update
        for epoch in range(epochs):
            old_p, old_v = net(obs)
            new_p, new_v = net(new)

            entropy = Categorical(old_p).entropy().mean()

            delta = (rew + gamma * new_v * (~don) - old_v).cpu().detach().numpy()

            advantage_list = []
            advantage = 0.0

            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])

            advantage_list.reverse()
            advantage = T.tensor(advantage_list, dtype=T.float32, device=device)

            tar_v = (advantage + old_v).detach()

            p_a = old_p.gather(-1, act)

            ratio = T.exp(T.log(p_a) - T.log(prb))

            surr1 = ratio * advantage
            surr2 = T.clamp(ratio, 1-cliprange, 1+cliprange) * advantage
            loss = -T.min(surr1, surr2) + smooth_l1_loss(old_v, tar_v) - entropy_coeff * entropy

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

        replay_buffer.reset()

        # Store policy to polciy buffer and local storage
        if (step + 1) % (horizon * save_interval) == 0:
            T.save(net.state_dict(), f'{save_path}/Models/{step + 1:08}.pt')

        if (step + 1) % (horizon * P_interval) == 0 and not use_builtin:
            policy_buffer.store_policy(net.state_dict())

    print()

