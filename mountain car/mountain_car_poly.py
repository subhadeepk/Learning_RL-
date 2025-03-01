import gymnasium as gym
import gym_simplegrid
import random 
import matplotlib.pyplot as plt
import numpy as np
# global Model_sampler_counter 
Model_sampler_counter = 0


# for _ in range(500):
#     print(env.step(2))

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
    
def update_w(episode, gamma, alpha, sarsa_n, w): #updates the Q[state][action] function at the end of each episode

    # print("in update w", w)

    episode_length = len(episode)

    if episode_length<sarsa_n: 
        return w
    
    # sarsa_n+=1

    current_position = episode[-1][0]
    current_velocity = episode[-1][1]
    current_action = episode[-1][2]
    current_state = np.array([[1, current_position,current_velocity, current_position * current_position, current_velocity * current_velocity, current_position * current_velocity]])

    # # Max Q value at that particular state, irrespective of which action is actually taken
    accumulated_return = max(max(Q(current_state, w)))
    

    # Finds accumulated return for sarsa_n != INF, basically bootstrapping:: sarsa_n very big for MC
    for time in range(episode_length-2 ,episode_length - sarsa_n -2, -1):

        state = np.array([[1, episode[time][0],episode[time][1], episode[time][0] ** 2,episode[time][1] ** 2, episode[time][0]*episode[time][1]]])
        action = int(episode[time][2])
        reward = episode[time][3]
        accumulated_return *= gamma
        G = reward + accumulated_return
        # print("Update Q", state, action, Q[state][action])

    
    #updating value of w
    grad_Q = state
    # print("state inside", state)
    w[action:action+1,:] += np.dot((alpha * (G - Q(state, w, action))) , grad_Q)
    # print("in update w", w)

    return w

def sample_state_reward_from_visited_states(Model):

    global Model_sampler_counter

    random_state = random.randint(0,63)
    random_action = random.randint(0,3)
    (next_state,reward) = Model[random_state][random_action]

    if Model_sampler_counter == 100:
        print("no samples found")
        Model_sampler_counter = 0
        return -1,-1,-1,-1
    elif (next_state,reward) != (-1,-1):
        Model_sampler_counter = 0
        return random_state,random_action,next_state,reward
    else:
        Model_sampler_counter +=1
        return sample_state_reward_from_visited_states(Model)
    
def get_feature_vector(pos,vel,no_of_features = 4):
    features = np.array[[1, pos, vel, pos*vel]]

# --------------Environment parameters------------------------------- 

# State-action Quality function Q

def Q(state, w, action=5):
    action = int(action)
    if action!=5:   
        w_ac = w[action:action+1, :]
        res = np.dot(w_ac,np.transpose(state))  #state is a 2*1 matrix, action is a 3*1 matrix and w is a 
        return(res)
    else:
        # print("inside Q", w, "tran", np.transpose(state))
        return np.dot(w,np.transpose(state))
    

env = gym.make('MountainCar-v0')
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 30)    # Between -1.2 and 0.6
vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 30)    # Between -0.07 and 0.07


#---------------------Learning parameters---------------------------------------
# Generate 500 episode, 500 max steps for each episode 
Learning_steps = 500 #Number of episodes
Planning_steps = 0
MAX_STEPS = 1000
Epsilon = 0.1  #for epsilon greedy policy: Chooses random action with epsilon probability
gamma = 0.95    #  discount factor
alpha = 0.1  #  step size 
sarsa_n = 1
num_actions = 3

w = np.zeros((num_actions,6)) #initializing the weight matrix

Performance_matrix = []



for i in range (0,Learning_steps):

    #generate episode 
    # st = i/4
    # if i%(M/5)==0:
    if i>Learning_steps-3:
        env = gym.make("MountainCar-v0", render_mode="human", goal_velocity=0.1)  # default goal_velocity=0
        epsilon = 0
        print("Episode No:", i+1)
    else:
        env = gym.make("MountainCar-v0", render_mode=None, goal_velocity=0.1)  # default goal_velocity=0
        epsilon = Epsilon




    episode = []
    obs = env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})
    obs, reward, done, _, info = env.step(1)
    done = False
    # print("Episode number", i+1)

    for steps_in_episode in range(MAX_STEPS):
        if done:
            break
        # env.step(2)
        # pos_state = np.digitize(obs[0], pos_space)
        # vel_state = np.digitize(obs[1], vel_space)
        pos_state = obs[0]
        vel_state = obs[1]
        state = np.array([[1, pos_state, vel_state, pos_state*pos_state, vel_state*vel_state, pos_state*vel_state]])

        # print(state, "w=", w)
        action_vals = Q(state,w)
        action = np.argmax(action_vals)

        obs_, reward, done, _, info = env.step(action)
        episode.append([pos_state, vel_state, action, reward])
        w = update_w(episode, gamma, alpha, sarsa_n, w)
        # print(obs_prev[0], obs_prev[1], action, reward)
        
    # # print(steps_in_episode)
    # # if i>2:
    # Performance_matrix.append((i, steps_in_episode))
    
    # print("updating Q")
    
    
    # env.close()
