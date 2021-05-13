# Imports
import torch as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.nn.functional import smooth_l1_loss, binary_cross_entropy

import json
import numpy as np

from Algos.Model.THGAIL import Model, Discrim
from Algos.utils import ReplayBufferTHGAIL, PolicyBuffer

def train(env, use_cuda, save_path):

    # Loading Config
    with open('Configs/config-THGAIL.json') as f: config = json.load(f)

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
    BC_epochs      = config['BC epochs']          # ENum epochs to run BC
    BC_batch_size  = config['BC batch size']      # Batch size used in BC

    GAIL_lr        = config['GAIL learning rate'] # Learing rate of the Discriminator
    GAIL_timestep  = config['GAIL timestep']      # Total number of timestpes to run GAIL

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
        from Algos.BC import trainP
        trainP(net, optim, device, BC_epochs, BC_batch_size)

    # Load data needed for GAIL and Setup net & optim
    GAIL_net   = Discrim().to(device)
    GAIL_optim = Adam(GAIL_net.parameters(), lr=GAIL_lr)

    demos = T.from_numpy(np.load('Data/GAIL/demos.npy').astype('float32')).to(device)
    demo_loader = DataLoader(demos, batch_size=horizon, shuffle=True)
    demo_iter = iter(demo_loader)

    # Set starting opponent equal to agent
    opp.load_state_dict(net.state_dict())

    # Final touches before self-play starts
    policy_buffer = PolicyBuffer(P_capacity)

    policy_buffer.store_policy(net.state_dict())

    replay_buffer = ReplayBufferTHGAIL(buffer_size=horizon, obs_dim=12, device=device)

    game_start = 0
    isPlayer2Serve = False

    observation = T.tensor(env.reset(not isPlayer2Serve), dtype=T.float32, device=device)

    # Main Loop
    for step in range(steps):

        with T.no_grad():
            p1, *_ = opp(observation[0])
            p2, *_ = net(observation[1])

        a1 = Categorical(p1).sample()
        a2 = Categorical(p2).sample()

        action = T.stack((a1, a2), dim=-1).cpu().detach()

        new_observation, reward, done, _ = env.step(action)

        new_observation = T.tensor(new_observation, dtype=T.float32, device=device)
        observation_action = T.cat([observation[1], action[1].unsqueeze(-1)])
        with T.no_grad(): irl_reward = -T.log(GAIL_net(observation_action)[0])
        reward = T.tensor(reward_coeff * reward, dtype=T.float32, device=device)
        done = T.tensor(done, dtype=T.bool, device=device)

        replay_buffer.store_data(observation[1], a2, irl_reward, reward, new_observation[1], p2[a2], done)

        observation = new_observation

        if done:
            observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32, device=device)

            isPlayer2Serve = not isPlayer2Serve

            print('\r', end=' ' * 80)
            print(f'\rCurrent Frame: {step+1}\tGame Length: {step-game_start}', end='')

            # Opponent Sampling (delta-Limit Uniform)
            if isPlayer2Serve:
                opp.load_state_dict(policy_buffer.sample())

            game_start = step

        if not replay_buffer.isFull:
            continue
        
        # Sample Data
        obs, act, new, irl, rew, prb, don = replay_buffer.sample()

        # PPO Update
        for epoch in range(epochs):
            old_p, old_i, old_v = net(obs)
            new_p, new_i, new_v = net(new)

            entropy = Categorical(old_p).entropy().mean()

            delta = (rew + gamma * new_v * (~don) - old_v).cpu().detach().numpy()

            advantage_list = []
            advantage_v = 0.0

            for delta_t in delta[::-1]:
                advantage_v = gamma * lmbda * advantage_v + delta_t[0]
                advantage_list.append([advantage_v])

            advantage_list.reverse()
            advantage_v = T.tensor(advantage_list, dtype=T.float32, device=device)

            tar_v = (advantage_v + old_v).detach()

            p_a = old_p.gather(-1, act)

            ratio = T.exp(T.log(p_a) - T.log(prb))

            surr1_v = ratio * advantage_v
            surr2_v = T.clamp(ratio, 1-cliprange, 1+cliprange) * advantage_v

            loss = smooth_l1_loss(old_v, tar_v) - entropy_coeff * entropy
            loss_p = -T.min(surr1_v, surr2_v)

            # GAIL Update
            if step < GAIL_timestep:
                try:
                    demo = demo_iter.next()
                except StopIteration:
                    demo_iter = iter(demo_loader)
                    demo = demo_iter.next()

                learner = GAIL_net(T.cat([obs, act], dim=1))
                expert  = GAIL_net(demo)

                loss_gail = binary_cross_entropy(learner, T.ones_like(learner, device=device)) +\
                       binary_cross_entropy(expert, T.zeros_like(expert, device=device))

                GAIL_optim.zero_grad()
                loss_gail.backward()
                GAIL_optim.step()

                alpha = step / GAIL_timestep

                delta = (irl + gamma * new_i * (~don) - old_i).cpu().detach().numpy()

                advantage_list = []
                advantage_i = 0.0

                for delta_t in delta[::-1]:
                    advantage_i = gamma * lmbda * advantage_i + delta_t[0]
                    advantage_list.append([advantage_i])

                advantage_list.reverse()
                advantage_i = T.tensor(advantage_list, dtype=T.float32, device=device)

                tar_i = (advantage_i + old_i).detach()

                surr1_i = ratio * advantage_i
                surr2_i = T.clamp(ratio, 1-cliprange, 1+cliprange) * advantage_i

                loss += smooth_l1_loss(old_i, tar_i)

                loss_p = alpha * loss_p + (1 - alpha) * (-T.min(surr1_i, surr2_i))

            loss += loss_p.mean()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

        replay_buffer.reset()

        # Store policy to polciy buffer and local storage
        if (step + 1) % (horizon * save_interval) == 0:
            T.save(net.state_dict(), f'{save_path}/Models/{step + 1:08}.pt')

        if (step + 1) % (horizon * P_interval) == 0:
            policy_buffer.store_policy(net.state_dict())

    print()
