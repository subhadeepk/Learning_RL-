import gymnasium as gym
import gym_simplegrid
import random 
import matplotlib.pyplot as plt
import numpy as np
import tiles3 as til
from itertools import product


# global Model_sampler_counter 
Model_sampler_counter = 0
global pos_prototypes, vel_prototypes, distance_scalar, pos_range, vel_range



# for _ in range(500):
#     print(env.step(2))

# global Model_sampler_counter 
Model_sampler_counter = 0

  
def update_w(episode, gamma, alpha, sarsa_n, w): #updates the Q[state][action] function at the end of each episode

    episode_length = len(episode)

    if episode_length<sarsa_n: 
        return w
    
    # sarsa_n+=1

    current_position = episode[-1][0]
    current_velocity = episode[-1][1]
    current_action = episode[-1][2]
    current_state = get_radial_basis_vector(current_position, current_velocity)

    # # Max Q value at that particular state, irrespective of which action is actually taken
    # accumulated_return = max(Q[current_state])

    # for n-step sarsa
    accumulated_return = max(max(Q(current_state, w)))
    # print(accumulated_return)


    # Finds accumulated return for sarsa_n != INF, basically bootstrapping:: sarsa_n very big for MC
    for time in range(episode_length-2 ,episode_length - sarsa_n -2, -1):

        state = get_radial_basis_vector(episode[time][0],episode[time][1])
        action = int(episode[time][2])
        reward = episode[time][3]
        accumulated_return *= gamma
        G = reward + accumulated_return
        # print("Update Q", state, action, Q[state][action])

    
    #updating value of w
    # print(Q(state, w, action))
    w[action,:] += (alpha * (G - Q(state, w, action))) * state
    return w
 
def get_radial_basis_vector(pos, vel):
    state = np.array([pos,vel])
    del_dist = (state - proto_points) / (np.array([pos_range, vel_range]) * distance_scalar)
    dist_sqr = np.sum(del_dist**2, axis=1)
    radial_basis = np.exp(-dist_sqr)
    normalized_rbf = radial_basis / np.sum(radial_basis)
    return normalized_rbf
    
def Q(state, w, action=5):
    action = int(action)
    if action!=5:   
        res = np.dot(w[action,:],state)
        return res
    else:
        all_action_qual = np.array([[np.dot(w[0,:],state),np.dot(w[1,:],state),np.dot(w[2,:],state)]])
        return all_action_qual
    

# --------------Environment parameters------------------------------- 
env = gym.make('MountainCar-v0')

#---------------------Learning parameters---------------------------------------
# Generate 500 episode, 500 max steps for each episode 
Learning_steps = 500 #Number of episodes
Planning_steps = 0
MAX_STEPS = 1000
Epsilon = 0.1  #for epsilon greedy policy: Chooses random action with epsilon probability
gamma = 0.95    #  discount factor
alpha = 0.3  #  learning rate 
sarsa_n = 4
no_of_prototypes_per_axis = 30
num_actions = 3
distance_scalar = 0.005 #controls the spread of the rbf  

# There are 8 tilings with 8 tiles on each axes, so the maximum index returned by tiling can be 8*8 = 64

w = np.zeros((num_actions, no_of_prototypes_per_axis**2)) #initializing the weight matrix
pos_prototypes = np.linspace(env.observation_space.low[0], env.observation_space.high[0], no_of_prototypes_per_axis) 
vel_prototypes = np.linspace(env.observation_space.low[1], env.observation_space.high[1], no_of_prototypes_per_axis) 
proto_points = np.array(list(product(pos_prototypes, vel_prototypes)))
pos_range = env.observation_space.high[0] - env.observation_space.low[0] 
vel_range = env.observation_space.high[1] - env.observation_space.low[1] 
Performance_matrix = []



for i in range (0,Learning_steps):

    #generate episode 
    # st = i/4
    # if i%(M/5)==0:
    if i>Learning_steps-3:
        env = gym.make("MountainCar-v0", render_mode="human", goal_velocity=0)  # default goal_velocity=0
        epsilon = 0
        print("Episode No:", i+1)
    else:
        env = gym.make("MountainCar-v0", render_mode=None, goal_velocity=0)  # default goal_velocity=0
        epsilon = Epsilon

    # print("Episode No:", i+1)



    episode = []
    obs = env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})
    obs, reward, done, _, info = env.step(1)
    done = False
    # print("Episode number", i+1)

    for steps_in_episode in range(MAX_STEPS):
        if done:
            break

    # while not done:
        # env.step(2)

        state = get_radial_basis_vector(obs[0], obs[1])
        # if i>Learning_steps-3:
        #     print(obs)
        #     print(state)
        action_vals = Q(state,w)
        action = np.argmax(action_vals)

        obs, reward, done, _, info = env.step(action)
        episode.append([obs[0], obs[1], action, reward])
        update_w(episode, gamma, alpha, sarsa_n, w)
        # print(obs_prev[0], obs_prev[1], action, reward)
        
    Performance_matrix.append((i, steps_in_episode))
    
    # print("updating Q")
    
    
# =================Performance metrics==========================================

x, y = zip(*Performance_matrix)

# Plot the graph
plt.gca().invert_yaxis()
plt.figure(figsize=(10, 6))

plt.plot(x, y, linestyle='-', color='b', label='Graph Line')

# Add labels and title
plt.xlabel('Learning step')
plt.ylabel('Steps taken to finish spisode')
plt.title(f'Q learning, 8 tilings of 8*8 tiles, learning rate = {alpha}, disc factor = {gamma}')
plt.legend()

plt.axhline(0, color='black', linewidth=0.8)  # Horizontal axis
plt.axvline(0, color='black', linewidth=0.8)  # Vertical axis
plt.grid(True, linestyle='--', alpha=0.7)     # Add gridlines for better visibility

plt.grid(visible=True)

# Show the graph and block program termination
plt.show(block=True)

from matplotlib.colors import ListedColormap

def get_action(x, y):
    """Function that returns 0, 1, or 2 based on x, y coordinates."""
    state = get_radial_basis_vector(x, y)
    action_vals = Q(state,w)
    action = np.argmax(action_vals)
    return action

def plot_grid(x_min, x_max, y_min, y_max, m, n):
    """Generates an m x n grid and colors each block based on get_action()."""
    x_vals = np.linspace(x_min, x_max, m)
    y_vals = np.linspace(y_min, y_max, n)
    grid_colors = np.zeros((len(y_vals), len(x_vals)))
    
    for i, y in enumerate(reversed(y_vals)):
        for j, x in enumerate(x_vals):
            grid_colors[i, j] = get_action(x, y)
    
    # Define color map using integers
    cmap = ListedColormap(['white', 'lightblue', 'darkblue'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(len(x_vals)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(y_vals)) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Use the colormap with numerical data
    ax.imshow(grid_colors, cmap=cmap, aspect='auto', extent=[x_min, x_max, y_min, y_max])
    plt.show()


plot_grid(-1.2, 0.6, -0.07, 0.07, 1000, 1000)


env.close()
    
