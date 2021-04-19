import os
import re
import gym
import argparse
import torch as T

import pyglet
from pyglet.window import key as keycodes

class Player:
    def __init__(self, p, env):
        self.p = p

        # 0: Human
        # 1: BuiltIn AI
        # 2: Trained AI
        self.playerType = 2

        if self.p == 'human':
            self.key_handler = pyglet.window.key.KeyStateHandler()
            self.enter_down = False
            env.viewer.window.push_handlers(self.key_handler)
            self.playerType = 0

        elif self.p == 'BuiltIn': 
            self.playerType = 1

        else:
            pattern = re.compile(r'\d\d-\d\d\d\d\d\d\d\d')
            matches = pattern.findall(p)

            if not (len(matches) == 1 and len(p) == 11):
                raise ValueError(f'{p} is incorrectly formated')

            run, version = matches[0].split('-')
            algo = __import__(f'Saves.Run{run}.Model', fromlist=[None])
            self.net = algo.Model()
            self.net.load_state_dict(T.load(f'Saves/Run{run}/Models/{version}.pt'))

    def sample(self, observation):
        if self.playerType == 0: return self.keyboard()
        if self.playerType == 1: return 0

        with T.no_grad(): p, *_ = self.net(observation)
        return T.distributions.Categorical(p).sample()

    def keyboard(self):
        action = 7

        keys_pressed = set()

        for key_code, pressed in self.key_handler.items():
            if pressed: keys_pressed.add(key_code)

        keys = set()

        for keycode in keys_pressed:
            for name in dir(keycodes):
                if getattr(keycodes, name) == keycode: keys.add(name)

        if 'LEFT' in keys: action -= 6
        elif 'RIGHT' in keys: action += 6

        if 'UP' in keys: action -= 1
        elif 'DOWN' in keys: action += 1

        if 'ENTER' in keys:
            if not self.enter_down: action += 3
            self.enter_down = True

        else: self.enter_down = False

        return action

    def __call__(self, *args):
        return self.sample(*args)

def Versus(pa, pb):
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0',
            isPlayer1Computer=pa == 'BuiltIn', 
            isPlayer2Computer=pb == 'BuiltIn')

    isPlayer2Serve = True

    observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

    pa = Player(pa, env)
    pb = Player(pb, env)

    while True:
        env.render()

        a1 = pa(observation[0])
        a2 = pb(observation[1])

        observation, reward, done, _ = env.step((a1, a2))

        observation = T.tensor(observation, dtype=T.float32)

        if done: 
            isPlayer2Serve = not isPlayer2Serve
            print(reward)
            observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pa', type=str, default='BuiltIn', help='Player on the left')
    parser.add_argument('--pb', type=str, default='BuiltIn', help='Player on the right')

    args = parser.parse_args()
    Versus(args.pa, args.pb)

