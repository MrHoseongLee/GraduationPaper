# Imports
import torch as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.nn.functional import smooth_l1_loss, binary_cross_entropy

import json
import numpy as np

from Algos.Model import PPO, Discrim
from Algos.utils import ReplayBufferPPO, PolicyBuffer

def train(env, use_cuda, save_path):

    # Loading Config
    with open('Configs/config-GAIL.json') as f: config = json.load(f)

    # PPO Config
    lr             = config['learning rate']      # Learing rate of the Actor/Critic
    steps          = config['total timestep']     # Total number of timesteps to train
    lmbda          = config['lambda']             # Lambda for GAE
    gamma          = config['gamma']              # Discount Factor
    epochs         = config['epochs']             # Num epochs for ppo update
    horizon        = config['horizon']            # Num steps before ppo update
    cliprange      = config['cliprange']          # Cliprange -> [0, 1]
    past_percent   = config['past percent']       # Percentage of past opp when sampling
    entropy_coeff  = config['entropy coeff']      # Entropy Coefficient
    save_interval  = config['save interval']      # Num steps before saving

    # BC Config
    use_BC         = config['use BC']             # Pretrain using BC or not
    BC_epochs      = config['BC epochs']          # ENum epochs to run BC
    BC_batch_size  = config['BC batch size']      # Batch size used in BC

    GAIL_lr        = config['GAIL learning rate'] # Learing rate of the Discriminator
    GAIL_timestep  = config['GAIL timestep']      # Total number of timestpes to run GAIL

    # Set device used for training
    device = T.device('cuda') if use_cuda else T.device('cpu')

    # Create the learning and opposing agents
    opp = PPO()
    net = PPO()

    opp.to(device)
    net.to(device)

    optim = Adam(net.parameters(), lr=lr)
    
    # Pretrain using Behavior Cloning
    if use_BC: 
        from Algos.BC import trainP
        trainP(net, optim, device, BC_epochs, BC_batch_size)

    # Load data needed for GAIL and Setup net & optim
    GAIL_net   = Discrim().to(device)
    GAIL_optim = Adam(GAIL_net.parameters(), lr=GAIL_lr)

    demos = T.from_numpy(np.load('Data/GAIL/demos.npy').astype('float32')).to(device)
    demo_loader = DataLoader(demos, batch_size=horizon, shuffle=True)

    # Set starting opponent equal to agent
    opp.load_state_dict(net.state_dict())

    # Final touches before self-play starts
    policy_buffer = PolicyBuffer()

    policy_buffer.store_policy(net.state_dict())

    replay_buffer = ReplayBufferPPO(buffer_size=horizon, obs_dim=12, device=device)

    observation = T.tensor(env.reset(), dtype=T.float32, device=device)

    game_start = 0
    oppIdx = -1

    # Main Loop
    for step in range(steps):

        with T.no_grad():
            p1, v1 = opp(observation[0])
            p2, v2 = net(observation[1])

        p1 = T.softmax(p1, dim=-1)
        p2 = T.softmax(p2, dim=-1)

        a1 = Categorical(p1).sample()
        a2 = Categorical(p2).sample()

        action = T.stack((a1, a2), dim=-1).cpu().detach().numpy()

        new_observation, reward, done, _ = env.step(action)

        new_observation = T.tensor(new_observation, dtype=T.float32, device=device)
        observation_action = T.cat([observation[1], action.unsqueeze(-1)])
        with T.no_grad(): irl_reward = -T.log(GAIL_net(observation_action)[0])
        done = T.tensor(done, dtype=T.bool, device=device)

        replay_buffer.store_data(observation[1], a2, irl_reward, new_observation[1], p2[a2], done)

        observation = new_observation

        if done:
            observation = T.tensor(env.reset(), dtype=T.float32, device=device)

            # Opponent Sampling
            if oppIdx >= 0:
                policy_buffer.store_result(oppIdx, reward.cpu().item() < 0)

            if np.random.random() < past_percent:
                oppIdx = policy_buffer.sample()
                opp.load_state_dict(policy_buffer.policies[oppIdx])
            else:
                oppIdx = -1
                opp.load_state_dict(net.state_dict())

            print('\r', end=' ' * 80)
            print(f'\rCurrent Frame: {step+1}\tGame Length: {step-game_start}', end='')

            game_start = step

        if not replay_buffer.isFull:
            continue
        
        # Sample Data
        obs, act, new, rew, prb, don = replay_buffer.sample()

        # GAIL Update
        demo = demo_loader.next()
        indices = T.randperm(len(demos))[:horizon]

        learner = GAIL_net(T.cat([obs, act], dim=1))
        expert  = GAIL_net(demos[indices])

        loss = binary_cross_entropy(learner, T.ones_like(learner, device=device)) +\
               binary_cross_entropy(expert, T.zeros_like(expert, device=device))

        GAIL_optim.zero_grad()
        loss.backward()
        GAIL_optim.step()

        # PPO Update
        for epoch in range(epochs):
            old_p, old_v = net(obs)
            new_p, new_v = net(new)

            old_p = T.softmax(old_p, dim=-1)
            new_p = T.softmax(new_p, dim=-1)

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
            policy_buffer.store_policy(net.state_dict())
            np.save(f'{save_path}/qs.npy', np.array(policy_buffer.qs))
            T.save(net.state_dict(), f'{save_path}/Models/PPO-{step + 1:08}.pt')

    print()
