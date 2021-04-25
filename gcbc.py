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

from model.py import make_model
from expert.py import BFS, get_expert_trajs
from four_rooms.py import FourRooms

def action_to_one_hot(env, action):
    action_vec = np.zeros(env.action_space.n)
    action_vec[action] = 1
    return action_vec  

class GCBC:

    def __init__(self, env, expert_trajs, expert_actions, random_goals = None):
        self.env = env
        self.expert_trajs = expert_trajs
        self.expert_actions = expert_actions
        self.transition_num = sum(map(len, expert_actions))
        self.model = make_model(input_dim=4, out_dim=4)
        # state_dim + goal_dim = 4
        # action_choices = 4

    def reset_model(self):
        self.model = make_model(input_dim=4, out_dim=4)	

    def generate_behavior_cloning_data(self):
        # 3 you will use action_to_one_hot() to convert scalar to vector
        # state should include goal
        self._train_states = []
        self._train_actions = []
        
        for _ in range(500):
            states, actions, goal = BFS(self.env)

            for i in range(len(states)):
              states[i] = np.concatenate((states[i],np.array(goal)))
              actions[i] = action_to_one_hot(self.env,self.env.act_map[tuple(actions[i].tolist())])
            self._train_states.extend(states)
            self._train_actions.extend(actions)

        self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
        self._train_actions = np.array(self._train_actions) # size: (*, 4)
    
    def generate_relabel_data(self):
        # 4 apply expert relabelling trick
        self._train_states = []
        self._train_actions = []

        for _ in range(500):
          states, actions, goal = BFS(self.env)
          relabel_states = states + [goal]
          for i in range(len(states)-1):
            relabel_idx = np.random.choice(range(i+1, len(states)+1))
            states[i] = np.concatenate((states[i], relabel_states[relabel_idx]))
            actions[i] = action_to_one_hot(self.env,self.env.act_map[tuple(actions[i].tolist())])
          states[-1] = np.concatenate((states[-1],np.array(goal)))
          actions[-1] = action_to_one_hot(self.env,self.env.act_map[tuple(actions[-1].tolist())])

          self._train_states.extend(states)
          self._train_actions.extend(actions)

        self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
        self._train_actions = np.array(self._train_actions) # size: (*, 4)

    def train(self, num_epochs=20, batch_size=256):
    # """ 3
        # Trains the model on training data generated by the expert policy.
        # Args:
        #   num_epochs: number of epochs to train on the data generated by the expert.
        # 	batch_size
        # Return:
        #   loss: (float) final loss of the trained policy.
        #   acc: (float) final accuracy of the trained policy
        # """
      print("States ", self._train_states.shape)
      print("Actions ", self._train_actions.shape)
      hist = self.model.fit(self._train_states,
                          self._train_actions,
                          epochs=num_epochs,
                          batch_size=batch_size)

      return hist.history['loss'][-1], hist.history['accuracy'][-1]*100

def evaluate_gc(env, policy, n_episodes=50):
    succs = 0
    for _ in range(n_episodes):
        _,_,_,info = generate_gc_episode(env, policy.model)

        if info == 'succ':
            succs +=1

    succs /= n_episodes
    return succs

def generate_gc_episode(env, policy):
    """Collects one rollout from the policy in an environment. The environment
    should implement the OpenAI Gym interface. A rollout ends when done=True. The
    number of states and actions should be the same, so you should not include
    the final state when done=True.
    Args:
        env: an OpenAI Gym environment.
        policy: a keras model
    Returns:
    """
    done = False
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    while not done:
        action = np.argmax(policy(np.expand_dims(state,0)))
            # print("action shape ", action)
        state, reward, done, info = env.step(action)
        if not done:
            states.append(state)
        actions.append(action_to_one_hot(env,action))
        rewards.append(reward)

    return states, actions, rewards, info