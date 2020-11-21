import gym
env = gym.make('CartPole-v0')


#Simple Hard coded Policy function 
def hard_coded_policy(observation):
    return int(observation[3] > 0) 

for i_episode in range(300):
    observation = env.reset()
    reward = 0
    for t in range(100):
        env.render()
        print(f"Observation:{observation}  \tReward: {reward}")

        # take a random action 
        #action = env.action_space.sample()  

        # retain center of gravity 
        action = hard_coded_policy(observation) 

        
        # Step returns 4 vals, info is debugging information
        # Obs = { CART_X_POS, CART_VEL, POLE_ANGLE, POLE_VEL_TIP } 
        # Reward = 1 if pole angle < 12 otherwise 0
        # done = Returns true if simulation is finished 
        observation, reward, done, info = env.step(action)
        
        #If done is true, it means sim is over  
        if done: 
            print(f"Episode finished after {t+1} timesteps")

env.close()



''' Not so good ''' 
'''

    Why?
      - Does not generalize well
      - Unrealistic constraints (Perfect environment with no friction or obstacles 
      - Unable to adapt to unforeseen circumstances 
'''





















