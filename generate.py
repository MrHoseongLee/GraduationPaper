import os
import re
import gym
import argparse
import torch as T
import numpy as np

def Generate(p, episodes):
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=False, isPlayer2Computer=False)

    run, version = p.split('-')
    algo = __import__(f'Saves.Run{run}.Model', fromlist=[None])
    net = algo.Model()
    net.load_state_dict(T.load(f'Saves/Run{run}/Models/{version}.pt'))

    obs, act = [], []
    obsT, actT = [], []

    isPlayer2Serve = True

    observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

    episode = 0

    while episode < episodes:
        print(f'\rGenerating Episode: {episode+1}', end='')
        while True:
            with T.no_grad():
                p1, *_ = net(observation[0])
                p2, *_ = net(observation[1])

                a1 = T.distributions.Categorical(p1).sample()
                a2 = T.argmax(p2)

            obsT.append(observation.numpy()[1])

            actT.append(a2.item())

            observation, reward, done, _ = env.step((a1, a2))

            observation = T.tensor(observation, dtype=T.float32)

            if done: 
                if reward == 1: obs.extend(obsT); act.extend(actT); episode += 1;

                obsT, actT = [], []
                isPlayer2Serve = not isPlayer2Serve
                observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)
                break

    obs = np.array(obs, dtype=np.float32)
    act = np.array(act, dtype=np.int64)

    demos = np.concatenate((obs, np.expand_dims(act, axis=1)), axis=1).astype(np.float32)

    np.save('Data/PPO/obs.npy', obs)
    np.save('Data/PPO/act.npy', act)
    np.save('Data/GAIL/demos.npy', demos)

    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--p', type=str, default='', help='Policy to generate data')
    parser.add_argument('--num', type=int, default=10, help='Number of episodes to generate')

    args = parser.parse_args()
    Generate(args.p, args.num)

