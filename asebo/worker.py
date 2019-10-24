
import numpy as np
import gym
import roboschool

from policies import ToeplitzPolicy, LinearPolicy

def get_policy(params):

    if params['policy'] == "Toeplitz":
        return(ToeplitzPolicy(params))
    elif params['policy'] == "Linear":
        return(LinearPolicy(params))

class worker(object):
    
    def __init__(self, params, master, A, i, train=True):
        
        self.env = gym.make(params['env_name'])
        self.env.seed(0)
        
        self.v = A[i, :]

        self.shift = params['shift']
        
        params['zeros'] = True
        self.policy = get_policy(params)
        self.policy.steps = params['steps']
        
        self.policy.update(master.params)
        self.timesteps = 0
    
    def do_rollouts(self, seed=0, train=True):
        
        self.policy.update(self.v)
        up = self.rollout(seed, train)
        
        self.policy.update(-2 * self.v)
        down = self.rollout(seed, train)
        
        return(np.array([up, down]))
    
    def rollout(self, seed=0, train=True):
        self.env.seed(seed)
        state = self.env.reset()
        self.env._max_episode_steps = self.policy.steps
        total_reward = 0
        done = False
        while not done:
            action = self.policy.evaluate(state)
            action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
            action = action.reshape(len(action), )
            state, reward, done, _ = self.env.step(action)
            if train:
                reward -= self.shift
            total_reward += reward
            self.timesteps += 1
        return(total_reward)

