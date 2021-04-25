from collections import OrderedDict 
import gym
from gym import spaces
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import random

class FourRooms:
    def __init__(self, l=5, T=30):
      # '''
      # FourRooms Environment for pedagogic purposes
      #   Each room is a l*l square gridworld,
      #   connected by four narrow corridors,
      #   the center is at (l+1, l+1).
      #   There are two kinds of walls:
      #   - borders: x = 0 and 2*l+2 and y = 0 and 2*l+2
      #   - central walls
      #   T: maximum horizion of one episode
      #       should be larger than O(4*l)
      #   '''
        assert l % 2 == 1 and l >= 5
        self.l = l
        self.total_l = 2 * l + 3
        self.T = T

    # create a map: zeros (walls) and ones (valid grids)
        self.map = np.ones((self.total_l, self.total_l), dtype=np.bool)
        # build walls
        self.map[0, :] = False
        self.map[-1, :] = False
        self.map[:, 0] = False
        self.map[:, -1] = False
        self.map[l+1, [1,2,-3,-2]] = self.map[[1,2,-3,-2], l+1] = False
        self.map[l+1, l+1] = False

        # define action mapping (go right/up/left/down, counter-clockwise)
        # e.g [1, 0] means + 1 in x coordinate, no change in y coordinate hence
        # hence resulting in moving right
        self.act_set = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1]
        ], dtype=np.int)
        self.action_space = spaces.Discrete(4)

        # you may use self.act_map in search algorithm
        self.act_map = {}
        self.act_map[(1, 0)] = 0
        self.act_map[(0, 1)] = 1
        self.act_map[(-1, 0)] = 2
        self.act_map[(0, -1)] = 3

    def render_map(self):
        plt.imshow(self.map)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.savefig('p2_map.png',
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()

    def sample_sg(self):
        # sample s
        while True:
            s = [np.random.randint(self.total_l),
                np.random.randint(self.total_l)]
            if self.map[s[0], s[1]]:
                break

        # sample g
        while True:
            g = [np.random.randint(self.total_l),
                np.random.randint(self.total_l)]
            if self.map[g[0], g[1]] and \
                (s[0] != g[0] or s[1] != g[1]):
                break
        return s, g

    def reset(self, s=None, g=None):
        '''
        s: starting position, np.array((2,))
        g: goal, np.array((2,))
        return obs: np.cat(s, g)
        '''
        if s is None or g is None:
            s, g = self.sample_sg()
        else:
            assert 0 < s[0] < self.total_l - 1 and 0 < s[1] < self.total_l - 1
            assert 0 < g[0] < self.total_l - 1 and 0 < g[1] < self.total_l - 1
            assert (s[0] != g[0] or s[1] != g[1])
            assert self.map[s[0], s[1]] and self.map[g[0], g[1]]

        self.s = s
        self.g = g
        self.t = 1

        return self._obs()

    def step(self, a):
        # '''
        # a: action, a scalar
        # return obs, reward, done, info
        # - done: whether the state has reached the goal
        # - info: succ if the state has reached the goal, fail otherwise
        # '''
        assert self.action_space.contains(a)

        # WRITE CODE HERE
        # print("curr state",self.s)
        # # print("action ", self.act_set[a])
        done = False
        next_state = self.s+self.act_set[a]
        # print('next state', next_state)
        info = 'fail'
        if self.t == self.T:
            # print("done " , self.T)
            done = True
        elif next_state[0] == self.g[0] and next_state[1] == self.g[1]:
            done = True
            # print("goal")
            info = "succ"
            self.s = next_state
            self.t+=1
        elif not self.map[next_state[0],next_state[1]]:
            self.t+=1
            # print(self.map[next_state])
            # print("obstacle")
        else:
            self.t+=1
            # print("valid")
            self.s = next_state

        # END

        return self._obs(), 0.0, done, info

    def _obs(self):
        return np.concatenate([self.s, self.g])

def test():
    s = np.array([1, 1])
    g = np.array([2*l+1, 2*l+1])
    s = env.reset(s, g)
    done = False
    traj = [s]
    while not done:
        s, _, done, _ = env.step(env.action_space.sample())
        traj.append(s)
    traj = np.array(traj)

    def plot_traj(env, ax, traj, goal=None):
        traj_map = env.map.copy().astype(np.float)
        traj_map[traj[:, 0], traj[:, 1]] = 2 # visited states
        traj_map[traj[0, 0], traj[0, 1]] = 1.5 # starting state
        traj_map[traj[-1, 0], traj[-1, 1]] = 2.5 # ending state
        if goal is not None:
            traj_map[goal[0], goal[1]] = 3 # goal
        ax.imshow(traj_map)
        ax.set_xlabel('y')
        ax.set_label('x')

    ax = plt.subplot()
    plot_traj(env, ax, traj, g)
    plt.savefig('p2_random_traj.png', 
            bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

# build env
l, T = 5, 30
env = FourRooms(l, T)
### Visualize the map
env.render_map()