import os
import sys
import gym
import shutil
import argparse

env_name = 'gym_pikachu_volleyball:pikachu-volleyball-v0'

def train():
    algos = ('PPO', 'THGAIL', 'GAIL')

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, required=True, help='What reinforcement learning algorithm to run')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda or not')
    parser.add_argument('--builtin', action='store_true', default=False, help='Use the builtin computer as the opponent or not')

    args = parser.parse_args()

    algo_name = args.algo
    use_builtin = args.builtin

    if algo_name not in algos:
        print(f'Error: Algorithm {algo_name} is not implemented yet')
        sys.exit(0)

    env = gym.make(env_name, isPlayer1Computer=use_builtin, isPlayer2Computer=False)

    algo = __import__(f'Algos.{args.algo}', fromlist=[None])

    save_id = len(os.listdir('Saves/')) + 1

    os.mkdir(f'Saves/Run{save_id:02}')
    os.mkdir(f'Saves/Run{save_id:02}/Models')

    shutil.copyfile(f'Configs/config-{algo_name}.json', f'Saves/Run{save_id:02}/config-{algo_name}.json')
    shutil.copyfile(f'Algos/Model/{algo_name}.py', f'Saves/Run{save_id:02}/Model.py')

    algo.train(env, use_builtin, args.cuda, f'Saves/Run{save_id:02}/')

if __name__ == '__main__':
    train()

