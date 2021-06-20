from math import sin, pi

def fade(step, total_steps):
    return sin(pi * step / total_steps)
