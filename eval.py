import gym
import argparse
import itertools
import torch as T

from glob import glob

def BulitIn(net, path, STEP, STEP_MUL, END_IDX, BI_CNT):
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0',
            isPlayer1Computer=True, isPlayer2Computer=False)

    isPlayer2Serve = True

    observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

    elos = [1200] * ((END_IDX - STEP * STEP_MUL - 1) // STEP // STEP_MUL + 1)

    for idx, V in enumerate(range(STEP * STEP_MUL, END_IDX, STEP * STEP_MUL)):
        print(f'\rBulit In at version {V}', end='')

        net.load_state_dict(T.load(path + '{:08d}.pt'.format(V)))

        gameCount, frameCount = 0, 0

        while True:
            with T.no_grad(): 
                p, *_ = net(observation[1])

            a = T.argmax(p)

            observation, reward, done, _ = env.step((0, a))

            observation = T.tensor(observation, dtype=T.float32)

            frameCount += 1

            if frameCount >= 900: done = True; reward = 0;

            if done: 
                isPlayer2Serve = not isPlayer2Serve
                observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

                frameCount = 0; gameCount += 1;

                reward = (reward + 1) / 2

                _, elos[idx] = updateELO(1200, elos[idx], 1 - reward, reward)

                if gameCount >= BI_CNT: break

    env.close()
    print()
    return elos

def Eval(RS, END_IDX, STEP, STEP_MUL, BI_CNT):
    RS.sort()

    N, nets, paths = len(RS), [], []

    for R in RS:
        path = 'Saves/Run{:02d}/'.format(R)

        net = __import__('Saves.Run{:02d}.Model'.format(R), fromlist=[None]).Model()
        net.eval()

        path = path + 'Models/'

        nets.append(net)
        paths.append(path)
        elo = BulitIn(net, path, STEP, STEP_MUL, END_IDX, BI_CNT)

        with open(f'Results/ELO-{R}.txt', 'w') as f:
            for e in elo: f.write(f'{e}\n')

def updateELO(RA, RB, SA, SB):
    EA = 1 / (1 + 10 ** ((RB - RA) / 400))
    EB = 1 / (1 + 10 ** ((RA - RB) / 400))
    return RA + 32 * (SA - EA), RB + 32 * (SB - EB)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--runs', nargs='*', type=int)

    parser.add_argument('--end', type=int, default=int(5e7))
    parser.add_argument('--step', type=int, default=64000)
    parser.add_argument('--step_mul', type=int, default=1)
    parser.add_argument('--bi_cnt', type=int, default=100)

    args = parser.parse_args()
    Eval(args.runs, args.end, args.step, args.step_mul, args.bi_cnt)

