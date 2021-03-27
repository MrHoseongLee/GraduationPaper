import gym
import numpy as np

def load_data_GAIL():
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)

    observations, actions = [], []

    with open('actions.txt') as f:
        lines = f.readlines()

        for line in lines:
            seed, recording, reward = line.split(',')
    
            reward = reward.strip('\n')

            if reward != '1': continue

            env.seed(int(seed))
            observation = env.reset()

            for action in recording:
                action = ord(action) - 65

                observations.append(observation[1])
                actions.append(action)

                observation, reward, _, _ = env.step((0, action))

    observations = np.array(observations, dtype='float32')
    actions = np.array(actions, dtype='float32')

    demos = np.concatenate((observations, np.expand_dims(actions, axis=1)), axis=1)

    np.save('GAIL/demos.npy', demos)

    env.close()

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

            observation, reward, done, _ = env.step((0, action))

            if reward_str == '1':
                new.append(observation[1])
                rew.append(reward)
                don.append(done)

    obs = np.array(obs, dtype='float32')
    act = np.array(act, dtype='uint8')
    new = np.array(new, dtype='float32')
    rew = np.array(rew, dtype='float32')
    don = np.array(don, dtype='bool')

    np.save('PPO/obs.npy', obs)
    np.save('PPO/act.npy', act)
    np.save('PPO/new.npy', new)
    np.save('PPO/rew.npy', rew)
    np.save('PPO/don.npy', don)

    env.close()

if __name__ == '__main__':
    #load_data_GAIL()
    load_data_PPO()

