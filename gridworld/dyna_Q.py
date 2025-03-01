import gymnasium as gym
import gym_simplegrid
import random 
import matplotlib.pyplot as plt
# global Model_sampler_counter 
Model_sampler_counter = 0

def pick_epsilon_greedy_action(actions, epsilon): #takes in state and samples action using epsilon greedy policy 

    # print(actions)
    if len(set(actions)) == 1:  #in the beginning, all actions are have same quality, hence get picked randomly
        action = random.randint(0,3)
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

    current_state = int(episode[episode_length-1][0])
    current_action = int(episode[episode_length-1][1])

    # Max Q value at that particular state, irrespective of which action is actually taken
    accumulated_return = max(Q[current_state])
    
    # print(episode)
    for time in range(episode_length-2 ,episode_length - sarsa_n -2, -1):

        state = int(episode[time][0])
        action = int(episode[time][1])
        reward = episode[time][2]
        accumulated_return *= gamma
        G = reward + accumulated_return
        # print("Update Q", state, action, Q[state][action])

    #updating value of t-n state and action
    Q[state][action] += alpha * (G - Q[state][action])
        

    return Q

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
    
options ={
    'start_loc': 55,
    'goal_loc': (2,0)
    # goal_loc is not specified, so it will be randomly sampled
}

# --------------Environment parameters------------------------------- 

# env = gym.make('SimpleGrid-8x8-v0', render_mode=None)
# episodes = {}   #Dictionary to hold episode experience using naive policy
ncol = 8
nrow = 8
nstates = nrow * ncol
nactions = 4

# State-action Quality function Q
# initialize all actions for all states to be equally likely initially, hence 0.25
# Access action Q using Q[state][action] = quaity of taking that action in that state
# actions = 0: UP 1: DOWN 2: LEFT 3: RIGHT

Q = [[25 for _ in range(nactions)] for _ in range(nstates)] 
Model = [[(-1,-1) for _ in range(nactions)] for _ in range(nstates)] 
    
#---------------------Learning parameters---------------------------------------
# Generate 500 episode, 500 max steps for each episode 
Learning_steps = 50 #Number of episodes
Planning_steps = 10
MAX_STEPS = 600
Epsilon = 0.1  #for epsilon greedy policy: Chooses random action with epsilon probability
gamma = 0.95    #  discount factor
alpha = 0.1  #  step size 
sarsa_n = 1
Performance_matrix = []

for i in range (0,Learning_steps):

    #generate episode 
    # st = i/4
    # if i%(M/5)==0:
    if i>Learning_steps-3:
        env = gym.make('SimpleGrid-8x8-v0', render_mode='human')
        epsilon = 0
        print("Episode No:", i+1)
    else:
        env = gym.make('SimpleGrid-8x8-v0', render_mode=None)
        epsilon = Epsilon


    episode = []
    obs, info = env.reset(seed=1, options=options)
    done = env.unwrapped.done
    # print("Episode number", i+1)

    for steps_in_episode in range(MAX_STEPS):
        if done:
            break
        obs_prev = obs
        # print(obs_prev, Q[obs_prev], len(Q))
        Q = update_Q_learing(episode, Q, gamma, alpha, sarsa_n)
        action = pick_epsilon_greedy_action(Q[obs_prev],epsilon)    #Epsilon greedy policy used
        obs, reward, done, _, info = env.step(action)
        Model[obs_prev][action] = (obs,reward)

        for _ in range(Planning_steps):
            #sample random state action pair, and make sure that it has been visited
            random_state,random_action,next_state,reward = sample_state_reward_from_visited_states(Model) 
            if random_state == -1:
                break

            # 1 step Q planning
            Q[random_state][random_action] += alpha * (reward + gamma*max(Q[next_state]) - Q[random_state][random_action])

        
            



        
        # print(obs,reward, done)
        episode.append([obs_prev, int(action), reward])
        
    # print(steps_in_episode)
    # if i>2:
    Performance_matrix.append((i, steps_in_episode))
    
    # print("updating Q")
    
    
    # env.close()

# Estimate values for each of the 64*4 state action pairs, as many times as there are episodes

# print(episodes)
# print(Performance_matrix)
x, y = zip(*Performance_matrix)
print(x)
print(y)

# Plot the graph
plt.gca().invert_yaxis()
plt.figure(figsize=(10, 6))

plt.plot(x, y, linestyle='-', color='b', label='Graph Line')

# Add labels and title
plt.xlabel('Learning step')
plt.ylabel('Steps taken to finish spisode')
plt.title(f'Performance of dyna_Q, #planning steps = {Planning_steps}, #Learning steps = {Learning_steps}')
plt.legend()

plt.axhline(0, color='black', linewidth=0.8)  # Horizontal axis
plt.axvline(0, color='black', linewidth=0.8)  # Vertical axis
plt.grid(True, linestyle='--', alpha=0.7)     # Add gridlines for better visibility

plt.grid(visible=True)

# Show the graph and block program termination
plt.show(block=True)




env.close()
