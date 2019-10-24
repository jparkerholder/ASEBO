
## External
import gym
import roboschool
import parser
import argparse
import numpy as np
import pandas as pd
import os

## Internal
from optimizers import Adam
from worker import worker, get_policy
from es import ES, aggregate_rollouts

def run_asebo(params):
        
    env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]
    
    m = 0
    v = 0

    params['k'] += -1
    params['alpha'] = 1
        
    params['zeros'] = False
    master = get_policy(params)
    
    if params['log']:
        params['num_sensings'] = 4 + int(3 * np.log(master.N))
    
    if params['k'] > master.N:
        params['k'] = master.N
        
    n_eps = 0
    n_iter = 1
    ts_cumulative = 0
    ts = []
    rollouts = []
    rewards = []
    samples = []
    alphas = []
    G = []
        
    while n_iter < params['max_iter']:
            
        params['n_iter'] = n_iter
        gradient, n_samples, timesteps = ES(params, master, G)
        ts_cumulative += timesteps
        ts.append(ts_cumulative)
        alphas.append(params['alpha'])

        if n_iter == 1:
            G = np.array(gradient)
        else:
            G *= params['decay']
            G = np.vstack([G, gradient])
        n_eps += 2 * n_samples
        rollouts.append(n_eps)
        gradient /= (np.linalg.norm(gradient) / master.N + 1e-8)
            
        update, m, v = Adam(gradient, m, v, params['learning_rate'], n_iter)
            
        master.update(update)
        test_policy = worker(params, master, np.zeros([1, master.N]), 0)
        reward = test_policy.rollout(train=False)
        rewards.append(reward)
        samples.append(n_samples)
            
        print('Iteration: %s, Rollouts: %s, Reward: %s, Alpha: %s, Samples: %s' %(n_iter, n_eps, reward, params['alpha'], n_samples))
        n_iter += 1
        
        out = pd.DataFrame({'Rollouts': rollouts, 'Reward': rewards, 'Samples': samples, 'Timesteps': ts, 'Alpha': alphas})
        out.to_csv('Seed%s.csv' %(params['seed']), index=False)        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Swimmer-v2')
    parser.add_argument('--steps', '-s', type=int, default=1000)
    parser.add_argument('--h_dim', '-hd', type=int, default=32)
    parser.add_argument('--start', '-st', type=int, default=0)
    parser.add_argument('--max_iter', '-it', type=int, default=1000)
    parser.add_argument('--seed', '-se', type=int, default=0)

    parser.add_argument('--k', '-k', type=int, default=70)
    parser.add_argument('--num_sensings', '-sn', type=int, default=100)
    parser.add_argument('--log', '-lg', type=int, default=0)
    parser.add_argument('--threshold', '-pc', type=float, default=0.995)
    parser.add_argument('--decay', '-dc', type=float, default=0.99)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.05)
    parser.add_argument('--filename', '-f', type=str, default='')
    parser.add_argument('--policy', '-po', type=str, default='Toeplitz') # Linear or Toeplitz

    parser.add_argument('--shift', '-sh', type=int, default=0)
    parser.add_argument('--min', '-mi', type=int, default=10)
    parser.add_argument('--sigma', '-si', type=float, default=0.1)

    args = parser.parse_args()
    params = vars(args)

    params['dir'] = params['env_name'] + params['policy'] + '_h' + str(params['h_dim']) + '_lr' + str(params['learning_rate']) + '_k' + str(params['k']) +'_' + params['filename']

    if not(os.path.exists('data/'+params['dir'])):
        os.makedirs('data/'+params['dir'])
    os.chdir('data/'+params['dir'])
    
    run_asebo(params)

