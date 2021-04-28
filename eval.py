import gym
import argparse
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

            if frameCount >= 1200: done = True; reward = 0.5;

            if done: 
                isPlayer2Serve = not isPlayer2Serve
                observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

                _, elos[idx] = updateELO(1200, elos[idx], 1 - reward, reward)

                frameCount = 0; gameCount += 1;

                if gameCount >= BI_CNT: break

    env.close()
    print()
    return elos

def Eval(RA, RB, END_IDX, STEP, STEP_MUL, BI_CNT, GAME_CNT):
    pathA = 'Saves/Run{:02d}/'.format(RA)
    pathB = 'Saves/Run{:02d}/'.format(RB)

    algo_nameA = glob(pathA + '*.json')[0].split('-')[-1].split('.')[0]
    algo_nameB = glob(pathB + '*.json')[0].split('-')[-1].split('.')[0]

    algoA = __import__('Saves.Run{:02d}.Model'.format(RA), fromlist=[None])
    algoB = __import__('Saves.Run{:02d}.Model'.format(RB), fromlist=[None])

    netA = algoA.Model()
    netB = algoB.Model()

    netA.eval()
    netB.eval()

    pathA += f'Models/'
    pathB += f'Models/'

    elosA = BulitIn(netA, pathA, STEP, STEP_MUL, END_IDX, BI_CNT)
    elosB = BulitIn(netB, pathB, STEP, STEP_MUL, END_IDX, BI_CNT)

    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0',
            isPlayer1Computer=False, isPlayer2Computer=False)

    isPlayer2Serve = True

    observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

    for idx, V in enumerate(range(STEP * STEP_MUL, END_IDX, STEP * STEP_MUL)):

        print(f'\rCompare at version {V}', end='')

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

            if frameCount >= 1200: done = True; reward = 0.5;

            if done: 
                isPlayer2Serve = not isPlayer2Serve
                observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

                frameCount = 0; gameCount += 1;

                elosA[idx], elosB[idx] = updateELO(elosA[idx], elosB[idx], 1 - reward, reward)

                if gameCount >= GAME_CNT: break

    print()

    with open('Results/ELO-{:02d}-{:02d}.txt'.format(RA, RB), 'w') as f:
        for eloA, eloB in zip(elosA, elosB):
            f.write(f'{eloA},{eloB}\n')
            
def updateELO(RA, RB, SA, SB):
    EA = 1 / (1 + 10 ** ((RB - RA) / 400))
    EB = 1 / (1 + 10 ** ((RA - RB) / 400))
    return RA + 32 * (SA - EA), RB + 32 * (SB - EB)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--runA', type=int)
    parser.add_argument('--runB', type=int)

    parser.add_argument('--end', type=int, default=int(1e8))
    parser.add_argument('--step', type=int, default=16000)
    parser.add_argument('--step_mul', type=int, default=10)
    parser.add_argument('--bi_cnt', type=int, default=100)
    parser.add_argument('--game_cnt', type=int, default=200)

    args = parser.parse_args()
    Eval(args.runA, args.runB, args.end, args.step, args.step_mul, args.bi_cnt, args.game_cnt)

