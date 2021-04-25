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
from queue import Queue

def BFS(env):
  curr_state, goal = env.sample_sg()
  q = Queue()
  curr_traj = [curr_state]
  curr_action = []
  seen = set({})
  seen.add((curr_state[0],curr_state[1]))
  q.put((curr_traj, curr_action, 0))
  
  min_dst=None
  expert_traj = None
  expert_action = None
  curr_dst=0
  while not q.empty():
    curr_traj, curr_action, curr_dst = q.get()
    curr_state = curr_traj[-1]
    
    if(curr_state[0] == goal[0] and curr_state[1] == goal[1]):
      if(expert_traj == None or curr_dst < min_dst):
        expert_traj = curr_traj[:-1]
        expert_action = curr_action
        min_dst =curr_dst

    for i in range(len(env.act_set)):
      action = env.act_set[i]
      next_state = curr_state + action
      
      # print("next ", type(next_state))
      if(env.map[next_state[0], next_state[1]]):
        if not ((next_state[0],next_state[1]) in seen):
          q.put((curr_traj + [next_state], curr_action + [action],curr_dst+1))
          seen.add((next_state[0],next_state[1]))

  assert(len(expert_traj) == len(expert_action))
  return expert_traj, expert_action, goal

def get_expert_trajs(N):
    expert_trajs = []
    expert_actions = []
    for _ in range(N):
        traj, actions,_ = BFS(env)
        expert_trajs += [np.array(traj)]
        expert_actions += [np.array(actions)]
    return expert_trajs, expert_actions

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

def test():
    fig, axes = plt.subplots(5,5, figsize=(10,10))
    axes = axes.reshape(-1)
    for idx, ax in enumerate(axes):
        plot_traj(env, ax, expert_trajs[idx])

    plt.savefig('p2_expert_trajs.png', 
            bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()