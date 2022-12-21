#%%
import pandas as pd
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

#%%
mg = Image.open('maze.jpg')
mg_raw = np.array(mg)
maze_raw = np.zeros([mg_raw.shape[0], mg_raw.shape[1]])
for i in range(mg_raw.shape[0]):
    for j in range(mg_raw.shape[1]):
        if mg_raw[i, j, :].sum() <= 100:
            maze_raw[i, j] = 1

maze = np.zeros([49, 65])
for i in range(49):
    for j in range(65):
        grid = maze_raw[8 * i + 48:8 * i + 56, 8 * j + 50:8 * j + 58]
        if grid.sum().sum() >= 55:
            maze[i, j] = 1
maze = np.concatenate([maze, np.zeros([1, 65])])
maze[-6, -2] = 1

def p_color(x):
    if x == 1:
        return np.array([255, 255, 255])
    elif x > 1:
        return np.array([255, 0, 0]) 
    else:
        return np.array([0, 0, 0])
mg2_raw = np.array([list(map(p_color, row)) for row in maze], dtype = 'uint8')
mg2 = Image.fromarray(mg2_raw)

#%%
direction_x = [-1, 0, 0, 1]
direction_y = [0, -1, 1, 0]
node_num = maze.shape[0] * maze.shape[1]
_state_num = len(np.where(maze > 0)[0])
_actions = dict()
num2tup = dict()
tup2num = dict()
n = -1
node_state = []
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        if maze[i, j]:
            n += 1
            num2tup[n] = (i, j)
            tup2num[(i, j)] = n
            node_state.append(n)
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        if maze[i, j]:
            _actions[tup2num[(i, j)]] = []
            for k in range(4):
                ii = i + direction_x[k]
                jj = j + direction_y[k]
                if maze[ii, jj]:
                    _actions[tup2num[(i, j)]].append(k)

np.random.seed(2)  
max_action_num = 4   
_epsilon = 0.95   # greedy police
_learning_rate = 0.1     # learning rate
_discount_factor = 0.9    # discount factor
max_episodes = 500   # maximum episodes                 

#%%
def build_q_table(_state_num, node_state, max_action_num):
    q_table = pd.DataFrame(
        np.zeros([_state_num, max_action_num]),
        columns = [0, 1, 2, 3],
        index = node_state
    )
    return q_table

def choose_action(state_current, q_table):
    action = None
    rand_num = np.random.uniform()
    candidate_actions = _actions[state_current]  
    if(rand_num > _epsilon):   
        ac_n = np.random.choice(candidate_actions)
    else:  
        max_q_value = -np.inf  
        for j in range(4):
            if(q_table.iloc[state_current, j] > max_q_value and j in candidate_actions):   
                max_q_value = q_table.iloc[state_current, j] 
                ac_n = j
    xx, yy = num2tup[state_current]
    x, y = xx + direction_x[ac_n], yy + direction_y[ac_n]
    action = tup2num[(x, y)]
    return ac_n, action

def get_env_feedback(state_current, action):
    arc = (state_current, action)
    if action == des:
        reward = 10000
    elif len(_actions[state_current]) == 1:
        reward = -100
    else:    
        reward = -1
    state_next = action
    return state_next, reward

def solve_spp_with_q_table(org_input, des_input, q_table):
    solution = [org_input]
    total_distance = 0
    current_node = org_input
    while current_node != des_input:
        ac_n = q_table.iloc[tup2num[current_node], :].argmax()
        xx, yy = current_node
        x, y = xx + direction_x[ac_n], yy + direction_y[ac_n]
        next_node = (x, y)
        solution.append(next_node)
        total_distance += 1
        current_node = next_node
    return solution, total_distance

def q_learning_algo():
    q_table = build_q_table(_state_num=_state_num, node_state=node_state, max_action_num=max_action_num)
    for episode in range(max_episodes):
        #if (episode % 10 == 0):
        print('enter episode: {}'.format(episode), end='  ')
        f_table = pd.DataFrame(False, index = q_table.index, columns = q_table.columns)
        state_current = org
        is_terminated = False
        #if (episode % 10 == 0):
        #print('current position: {}'.format(state_current), end='   ')
        #print('next position: ', end='')
        step_counter = 0
        while not is_terminated:
            ac_n, action = choose_action(state_current, q_table)
            state_next, reward = get_env_feedback(state_current, action)   
            f_table.iloc[state_current, ac_n] = True
            #if (episode % 10 == 0):
            #print(' {} '.format(state_next), end='')
            q_predict = q_table.iloc[state_current, ac_n]
            q_target = 0
            if(state_next != des):
                q_target = reward + _discount_factor * q_table.iloc[state_next, :].max()   
            else:
                q_target = reward      
                is_terminated = True   
            q_table.iloc[state_current, ac_n] += _learning_rate * (q_target - q_predict)  
            state_current = state_next  
            step_counter += 1
        #if (episode % 10 == 0):
        print(step_counter)
        for i in range(q_table.shape[0]):
            for j in range (q_table.shape[1]):
                if f_table.iloc[i, j]:
                    q_table.iloc[i, j] += 1
        #print(q_table, end='\n\n')
        #q_table.to_csv(str(episode) + '.csv')
    return q_table

#%%
org_input  = (44, 63)
des_input = (1, 1)
if des_input not in tup2num:
    print('Destination Error!')
else:
    org = tup2num[org_input]
    des = tup2num[des_input]

    q_table = build_q_table(_state_num, node_state, max_action_num)
    q_table = q_learning_algo()
    print('Final q_table: \n', q_table, end='\n\n')

    solution, total_distance = solve_spp_with_q_table(org_input, des_input, q_table)
    maze_ans = maze.copy()
    for i in range(len(solution)):
        x, y = solution[i]
        maze_ans[x, y] = 2
    mg3_raw = np.array([list(map(p_color, row)) for row in maze_ans], dtype = 'uint8')
    mg3 = Image.fromarray(mg3_raw)
    print(mg3)
    print(total_distance)
