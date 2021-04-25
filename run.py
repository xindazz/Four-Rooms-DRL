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
from gcbc.py import GCBC

N = 1000
expert_trajs, expert_actions = get_expert_trajs(N)

# build env
l, T = 5, 30
env = FourRooms(l, T)

gcbc = GCBC(env, expert_trajs, expert_actions)
# mode = 'vanilla'
mode = 'relabel'

# print(gcbc._train_states.shape)
# print(gcbc._train_actions.shape)
# print("Model", )
num_seeds = 5
loss_vecs = []
acc_vecs = []
succ_vecs = []

for i in range(num_seeds):
    print('*' * 50)
    print('seed: %d' % i)
    loss_vec = []
    acc_vec = []
    succ_vec = []
    gcbc.reset_model()

    for e in range(200):
        print("SEED ", i, " iter ", e)
        if mode =='vanilla':
            print("Generating Vanilla data")
            gcbc.generate_behavior_cloning_data()
        else:
            print("Generating Relabel data")
            gcbc.generate_relabel_data()
        
        loss, acc = gcbc.train(num_epochs=20)
        succ = evaluate_gc(env, gcbc)
        loss_vec.append(loss)
        acc_vec.append(acc)
        succ_vec.append(succ)
        print(e, round(loss,3), round(acc,3), succ)
    loss_vecs.append(loss_vec)
    acc_vecs.append(acc_vec)
    succ_vecs.append(succ_vec)

loss_vec = np.mean(np.array(loss_vecs), axis = 0).tolist()
acc_vec = np.mean(np.array(acc_vecs), axis = 0).tolist()
succ_vec = np.mean(np.array(succ_vecs), axis = 0).tolist()

### Plot the results
from scipy.ndimage import uniform_filter
# you may use uniform_filter(succ_vec, 5) to smooth succ_vec
succ_vec = uniform_filter(succ_vec, 5)
plt.figure(figsize=(12, 3))
# WRITE CODE HERE
plt.plot(loss_vec, label='loss')
plt.savefig('loss_p2_gcbc_%s.png' % mode, dpi=300)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
plt.figure(figsize=(12, 3))
plt.plot(np.array(acc_vec) / 100, label='accuracy')
plt.savefig('acc_p2_gcbc_%s.png' % mode, dpi=300)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
plt.figure(figsize=(12, 3))
plt.plot(succ_vec, label='success rate')
plt.savefig('succ_p2_gcbc_%s.png' % mode, dpi=300)
plt.xlabel("Iteration")
plt.ylabel("Success rate")
# plt.ylim([0,1])
plt.show()
# END