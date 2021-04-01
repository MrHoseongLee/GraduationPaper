import os
import gym
import argparse
import torch as T

END_IDX   = int(1e8)
STEP_SIZE = 16000 * 50
BI_GAME_COUNT = 10

def Compare(algo_nameA, algo_nameB, RA, RB):
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0',
            isPlayer1Computer=False, isPlayer2Computer=False)

    isPlayer2Serve = True

    observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

    algoA = __import__('Saves.Run{RA:02}.Model', fromlist=[None])
    algoB = __import__('Saves.Run{RB:02}.Model', fromlist=[None])

    netA = algoA.PPO()
    netB = algoB.PPO()

    netA.eval()
    netB.eval()

    with open(f'WinRate-{RA:02}-{RB:02}.txt', 'w') as f:
        for V in range(STEP_SIZE, END_IDX, STEP_SIZE):
            netA.load_state_dict(T.load(f'Saves/Run{RA:02}/Models/{algo_nameA}-{V:08}.pt'))
            netB.load_state_dict(T.load(f'Saves/Run{RB:02}/Models/{algo_nameB}-{V:08}.pt'))

            gameCount = 0
            frameCount = 0

            winA = 0
            drawA = 0
            lossA = 0

            while True:
                with T.no_grad(): 
                    p1, _ = netA(observation[0])
                    p2, _ = netB(observation[1])

                a1 = T.distributions.Categorical(p1).sample()
                a2 = T.distributions.Categorical(p2).sample()

                observation, reward, done, _ = env.step((a1, a2))

                observation = T.tensor(observation, dtype=T.float32)

                frameCount += 1

                if frameCount >= 1200: done = True; reward = 0;

                if done: 
                    isPlayer2Serve = not isPlayer2Serve
                    observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)
                    frameCount = 0
                    gameCount += 1

                    if reward < 0: winA += 1
                    if reward == 0: drawA += 1
                    if reward > 0: lossA += 1

                    if gameCount >= BI_GAME_COUNT: break
            
            f.write(f'{winA},{drawA},{lossA}\n')
            f.flush()
            os.fsync(f.fileno())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--algoA', type=str, default='PPO', help='What reinforcement learning algorithm to run')
    parser.add_argument('--algoB', type=str, default='PPO', help='What reinforcement learning algorithm to run')

    parser.add_argument('--runA', type=int, default='', help='')
    parser.add_argument('--runB', type=int, default='', help='')

    args = parser.parse_args()
    Compare(args.algo, args.runA, args.runB)

