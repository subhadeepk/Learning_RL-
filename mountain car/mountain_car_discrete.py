import gymnasium as gym
import gym_simplegrid
import random 
import matplotlib.pyplot as plt
import numpy as np
# global Model_sampler_counter 
Model_sampler_counter = 0

def pick_epsilon_greedy_action(actions, epsilon): #takes in state and samples action using epsilon greedy policy 

    # print(actions)
    if len(set(actions)) == 1:  #in the beginning, all actions are have same quality, hence get picked randomly
        action = random.randint(0,2)
        # print("uniform", action)
        return action
    
    max_index = actions.index(max(actions))  

    # Choose whether to return max_index or a random index
    if random.random() < epsilon:
        # Return a random non-max index
        non_max_indices = [i for i in range(len(actions)) if i != max_index]
        return random.choice(non_max_indices)
    else:
        # Return the index of the maximum element
        return max_index
    
def update_Q_learing(episode, Q, gamma, alpha, sarsa_n): #updates the Q[state][action] function at the end of each episode

    episode_length = len(episode)

    if episode_length<sarsa_n: 
        return Q
    
    # sarsa_n+=1

    current_state_pos = int(episode[-1][0])
    current_state_vel = int(episode[-1][1])
    current_action = int(episode[-1][2])

    # # Sarsa - n
    # accumulated_return = Q[current_state_pos, current_state_vel, current_action]

    #Q learning
    accumulated_return = max(Q[current_state_pos, current_state_vel])
    
    # print(episode)
    for time in range(episode_length-2 ,episode_length - sarsa_n -2, -1):

        state_pos = int(episode[time][0])
        state_vel = int(episode[time][1])
        action = int(episode[time][2])
        reward = episode[time][3]
        accumulated_return *= gamma
        G = reward + accumulated_return
        # print("Update Q", state, action, Q[state][action])

    #updating value of t-n state and action
    Q[state_pos, state_vel, action] += alpha * (G - Q[state_pos, state_vel, action])
        

    return Q

nactions = 3

env = gym.make('MountainCar-v0')
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 30)    # Between -1.2 and 0.6
vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 30)    # Between -0.07 and 0.07
# print(pos_space)
# print(vel_space)

Q = np.zeros((len(pos_space), len(vel_space), nactions)) #  20x20x3 array
# print(np.shape(Q))

    
#---------------------Learning parameters---------------------------------------
# Generate 500 episode, 500 max steps for each episode 
Learning_steps =3000 #Number of episodes
Planning_steps = 0
MAX_STEPS = 1000
Epsilon = 1  #for epsilon greedy policy: Chooses random action with epsilon probability
gamma = 0.9    #  discount factor
alpha = 0.9  #  step size 
sarsa_n = 2
Performance_matrix = []
epsilon_decay_rate = 2/Learning_steps

for i in range (0,Learning_steps):

    #generate episode 
    # st = i/4
    # if i%(M/5)==0:
    if i>Learning_steps-3:
        env = gym.make('MountainCar-v0', render_mode='human')
        # epsilon = 0
        print("Episode No:", i+1)
    else:
        env = gym.make('MountainCar-v0', render_mode=None)
        # epsilon = max(Epsilon - epsilon_decay_rate, 0)



    episode = []
    obs = env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})
    obs = [0, 0]
    done = False
    # print("Episode number", i+1)

    for steps_in_episode in range(MAX_STEPS):
        if done:
            break
        # env.step(2)
        
        Q = update_Q_learing(episode, Q, gamma, alpha, sarsa_n)
        pos_state = np.digitize(obs[0], pos_space)
        vel_state = np.digitize(obs[1], vel_space)

        # print("epsilon",epsilon)
        action = np.argmax(Q[pos_state, vel_state, :])#pick_epsilon_greedy_action(Q[pos_state, vel_state, :].tolist(), epsilon)

        obs, reward, done, _, info = env.step(action)
        episode.append([pos_state, vel_state, action, reward])
        # print(action, reward)





env.close()
