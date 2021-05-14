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

def Eval(RS, END_IDX, STEP, STEP_MUL, BI_CNT, GAME_CNT):
    RS.sort()
    eval_id = '-'.join('{:02d}'.format(R) for R in RS)

    N, nets, paths, elos = len(RS), [], [], []

    for R in RS:
        path = 'Saves/Run{:02d}/'.format(R)

        net = __import__('Saves.Run{:02d}.Model'.format(R), fromlist=[None]).Model()
        net.eval()

        path = path + 'Models/'

        nets.append(net)
        paths.append(path)
        elos.append(BulitIn(net, path, STEP, STEP_MUL, END_IDX, BI_CNT))

    with open(f'Results/ELOC-{eval_id}.txt', 'w') as f:
        for elo in elos: f.write(','.join(str(e) for e in elo) + '\n')

    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0',
            isPlayer1Computer=False, isPlayer2Computer=False)

    isPlayer2Serve = True

    observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

    for idx, V in enumerate(range(STEP * STEP_MUL, END_IDX, STEP * STEP_MUL)):
        print(f'\rCompare at version {V}', end='')

        elos_mean = [ 0 ] * N

        for a in range(N): 
            for b in range(a + 1, N):
                netA, pathA, eloA = nets[a], paths[a], elos[a][idx]
                netB, pathB, eloB = nets[b], paths[b], elos[a][idx]

                netA.load_state_dict(T.load(pathA + '{:08d}.pt'.format(V)))
                netB.load_state_dict(T.load(pathB + '{:08d}.pt'.format(V)))

                gameCount, frameCount = 0, 0

                while True:
                    with T.no_grad(): 
                        p1, *_ = netA(observation[0])
                        p2, *_ = netB(observation[1])

                    a1 = T.argmax(p1)
                    a2 = T.argmax(p2)

                    observation, reward, done, _ = env.step((a1, a2))

                    observation = T.tensor(observation, dtype=T.float32)

                    frameCount += 1

                    if frameCount >= 900: done = True; reward = 0;

                    if done: 
                        isPlayer2Serve = not isPlayer2Serve
                        observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

                        frameCount = 0; gameCount += 1;

                        reward = (reward + 1) / 2

                        elo = updateELO(eloA, eloB, 1 - reward, reward)

                        elos_mean[a] += elo[0]
                        elos_mean[b] += elo[1]

                        if gameCount >= GAME_CNT: break

        for i in range(N): elos[i][idx] = elos_mean[i] / (N - 1)

    print()

    with open(f'Results/ELOF-{eval_id}.txt', 'w') as f:
        for elo in elos: f.write(','.join(str(e) for e in elo) + '\n')
            
def updateELO(RA, RB, SA, SB):
    EA = 1 / (1 + 10 ** ((RB - RA) / 400))
    EB = 1 / (1 + 10 ** ((RA - RB) / 400))
    return RA + 32 * (SA - EA), RB + 32 * (SB - EB)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--runs', nargs='*', type=int)

    parser.add_argument('--end', type=int, default=int(5e8))
    parser.add_argument('--step', type=int, default=6400)
    parser.add_argument('--step_mul', type=int, default=10)
    parser.add_argument('--bi_cnt', type=int, default=100)
    parser.add_argument('--game_cnt', type=int, default=100)

    args = parser.parse_args()
    Eval(args.runs, args.end, args.step, args.step_mul, args.bi_cnt, args.game_cnt)

