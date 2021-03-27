import gym
import argparse
import torch as T
from torch.distributions import Categorical

import pyglet
from pyglet.window import key as keycodes

env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=False, isPlayer2Computer=False)
#env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)

env.reset()
env.render()

key_handler = pyglet.window.key.KeyStateHandler()
env.viewer.window.push_handlers(key_handler)

enter_down = False

def getInput():
    global enter_down

    action = [ 0 ] * 3

    keys_pressed = set()

    for key_code, pressed in key_handler.items():

        if pressed:
            keys_pressed.add(key_code)

    keys = set()

    for keycode in keys_pressed:
        for name in dir(keycodes):
            if getattr(keycodes, name) == keycode:
                keys.add(name)

    if 'LEFT' in keys:
        action[0] = -1
    elif 'RIGHT' in keys:
        action[0] = 1

    if 'UP' in keys:
        action[1] = -1
    elif 'DOWN' in keys:
        action[1] = 1

    if 'ENTER' in keys:
        if not enter_down:
            action[2] = 1

        enter_down = True
    else:
        enter_down = False

    return (action[0] + 1) * 6 + (action[1] + 1) + action[2] * 3

def test(algo_name, R, V, player):

    observation = T.tensor(env.reset(), dtype=T.float32)

    from Algos.Model import PPO

    net = PPO()
    net.load_state_dict(T.load(f'Saves/Run{R}/Models/{algo_name}-{V:08}.pt'))
    net.eval()

    while True:
        env.render()

        with T.no_grad(): p1, _ = net(observation[0])

        p1 = T.softmax(p1, dim=-1)

        a1 = Categorical(p1).sample()

        if player:
            a2 = T.tensor(getInput(), dtype=T.int64)

        else:
            with T.no_grad(): p2, _ = net(observation[1])

            p2 = T.softmax(p2, dim=-1)

            a2 = Categorical(p2).sample()

        #a1 = T.argmax(out1)
        #a2 = T.argmax(out2)

        action = T.stack((a1, a2), dim=-1).cpu().detach().numpy()

        observation, reward, done, _ = env.step(action)

        observation = T.tensor(observation, dtype=T.float32)

        if done:
            observation = T.tensor(env.reset(), dtype=T.float32)
            print(reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='PPO', help='What reinforcement learning algorithm to run')
    parser.add_argument('--player', action='store_true', default=False, help='Use human opponent or not')
    parser.add_argument('--run', type=str, default='', help='')
    parser.add_argument('--version', type=int, default='', help='What version of the trained model to run')
    args = parser.parse_args()
    test(args.algo, args.run, args.version, args.player)

