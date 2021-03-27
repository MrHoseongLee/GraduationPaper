import gym
import numpy as np

def load_data_PPO():
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)
    obs, act, new, rew, don = [], [], [], [], []

    with open('actions.txt') as f:
        lines = f.readlines()

    for line in lines:
        seed, recording, reward_str = line.split(',')

        reward_str = reward_str.strip('\n')

        env.seed(int(seed))
        observation = env.reset()

        for action in recording:
            action = ord(action) - 65

            if reward_str == '1':
                obs.append(observation[1])
                act.append(action)

            observation, _, _, _ = env.step((0, action))

    obs = np.array(obs, dtype='float32')
    act = np.array(act, dtype='uint8')

    np.save('PPO/obs.npy', obs)
    np.save('PPO/act.npy', act)

    env.close()

if __name__ == '__main__':
    load_data_PPO()

