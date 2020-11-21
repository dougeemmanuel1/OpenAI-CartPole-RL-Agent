import gym
from sklearn.preprocessing import KBinsDiscretizer
import math 
import numpy as np
import torch 
from typing import Tuple


env = gym.make('CartPole-v0')

# Divide state space into discrete buckets
n_bins = ( 6, 12 )
lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]


def discretizer( _ , __ , angle, pole_velocity) -> Tuple[int,...]:
    '''
        Convert continuous states into a discrete state for Q-Learning
    '''
    # Utilize Scikit-learns KBINsDiscretizer
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([ lower_bounds, upper_bounds ])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))


# Initialize Q value table with Zeros 
q_table = np.zeros(n_bins + (env.action_space.n,))
def policy( state : tuple):
    '''
        Create a policy function that uses the Q-Table to look up and greedily
        select highest Q value.
    '''
    return np.argmax(q_table[state])

def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
    '''
        Temporal diffrence for updating Q-value of state-action pair
    '''
    future_optimal_value = np.max(q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

# Adaptive learning of Learning Rate
def learning_rate(n : int , min_rate=0.01 ) -> float  :
    '''
        Decaying learning rate
    '''
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n : int, min_rate= 0.1 ) -> float :
    ''' 
        Decaying exploration rate
    '''
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))

num_episodes = 10000
for n in range(num_episodes):
    print(f"Episode : {n} of {num_episodes}") 
    #Discretize state into buckets using sci-kit learn
    current_state, done = discretizer(*env.reset()), False 

    #Run until we receive queue from the environment that sim is over 
    while not done:
        #get policy from action 
        action =  policy(current_state)
        
        # take a random action but use for exploration now 
        if np.random.random() < exploration_rate(n):
            action = env.action_space.sample() 
        
        # Step returns 4 vals, info is debugging information
        # Obs = { CART_X_POS, CART_VEL, POLE_ANGLE, POLE_VEL_TIP } 
        # Reward = 1 if pole angle < 12 otherwise 0
        # done = Returns true if simulation is finished 
        observation, reward, done, debug = env.step(action)
        new_state = discretizer(*observation)
    
        # Update Q-Table
        lr = learning_rate(n)
        learned_value = new_Q_value(reward , new_state )
        old_value = q_table[current_state][action]
        q_table[current_state][action] = (1-lr)*old_value + lr*learned_value
        
        current_state = new_state 
        
        #render the env
        env.render()

