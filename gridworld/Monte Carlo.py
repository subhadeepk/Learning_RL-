import gymnasium as gym
import gym_simplegrid
import random 

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
    

def update_Q(episode, Q, gamma, alpha): #updates the Q[state][action] function at the end of each episode
    episode_length = len(episode)
    # print("episode length =", episode_length)
    accumulated_return = 0
    # print(episode)
    for time in range(episode_length-1,-1, -1):
        # print(time)
        # print(episode[time])
        state = int(episode[time][0])
        action = int(episode[time][1])
        reward = episode[time][2]
        G = reward + accumulated_return
        accumulated_return = gamma*G
        # print("Update Q", state, action, Q[state][action])
        Q[state][action] += alpha * (G - Q[state][action])

    return Q

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

#---------------------Learning parameters---------------------------------------
# Generate M episode, MAX_STEPS max steps for each episode 
M =50000 #Number of episodes
MAX_STEPS = 1000
epsilon = 0.2   #for epsilon greedy policy: Chooses random action with epsilon probability
gamma = 1    #  discount factor
alpha = 1/500  #  step size 

for i in range (0,M):

    #generate episode 
    # if i%(M/5)==0:
    if i%5 == 0:
        env = gym.make('SimpleGrid-8x8-v0', render_mode='human')
        epsilon = 0
        print("Episode No:", i+1)
    else:
        env = gym.make('SimpleGrid-8x8-v0', render_mode=None)


    episode = []
    obs, info = env.reset(seed=1, options=options)
    done = env.unwrapped.done
    # print("Episode number", i+1)

    for _ in range(MAX_STEPS):
        if done:
            break
        obs_prev = obs
        # print(obs_prev, Q[obs_prev], len(Q))
        action = pick_epsilon_greedy_action(Q[obs_prev],epsilon)    #Epsilon greedy policy used

        obs, reward, done, _, info = env.step(action)
        # print(obs,reward, done)
        episode.append([obs_prev, int(action), reward])
    
    # print("updating Q")
    Q = update_Q(episode, Q, gamma, alpha)
    
    # env.close()

# Estimate values for each of the 64*4 state action pairs, as many times as there are episodes

# print(episodes)
env.close()

print("aveda  kedavra")