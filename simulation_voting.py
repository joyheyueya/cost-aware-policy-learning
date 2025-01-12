import numpy as np
import pandas as pd
from datetime import datetime
import random
import alg
import pickle
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--lam', default=1, type=float)
parser.add_argument('--train_file_name', default='voting_data/train_processed_n01.pkl', type=str)
parser.add_argument('--test_file_name', default='voting_data/test_processed_n01.pkl', type=str)
parser.add_argument('--results_file_name', default='results_voting/voting', type=str)
args = parser.parse_args()
if not os.path.exists(args.results_file_name.split('/')[0]):
    try:
        os.makedirs(args.results_file_name.split('/')[0])
    except:
        print('file already exists.')

with open(args.train_file_name, 'rb') as f:
    TRAIN_PROCESSED = pickle.load(f)
VAL_PROCESSED = TRAIN_PROCESSED
with open(args.test_file_name, 'rb') as f:
    TEST_PROCESSED = pickle.load(f)
d = TRAIN_PROCESSED[0][0].shape[1]
print('d', d)
print(TRAIN_PROCESSED[0][2])

def sample_offline(method, experiment_size):
    if method == 'S-P':
        sp_alg = alg.SP(args.lam, d)
    elif method == 'S-P_cost':
        sp_alg = alg.SP_cost(args.lam, d)
    elif method == 'S-P_cost_v2':
        sp_alg = alg.SP_cost(args.lam, d)
    results_df = pd.DataFrame()
    cs_list = []
    a_list = []
    vals_list = []
    costs_list = []
    while len(cs_list) <  experiment_size:
        print('offline', len(cs_list))
        print(datetime.now())
        for n in VAL_PROCESSED.keys():
            sample = VAL_PROCESSED[n]
            cs = sample[0]
            ys = sample[1]
            costs = sample[2]
            if method == 'S-P':
                a,vals = sp_alg.pi_offline(cs)
            elif method == 'S-P_cost':
                a,vals,c = sp_alg.pi_offline(cs, costs)
            elif method == 'S-P_cost_v2':
                a,_,_ = sp_alg.pi_offline(cs, None)
            if not np.isnan(ys[a]):
                sp_alg.update_offline(cs[a])

                cs_list.append(cs)
                a_list.append(a)
                vals_list.append(vals)
                costs_list.append(costs)

    return sp_alg

def learn(X, Y, tmp_lamb):
    X, Y = np.array(X), np.array(Y)
    d = X.shape[1]
    A = X.T.dot(X) + np.eye(d) * tmp_lamb
    b = X.T.dot(Y)
    w = np.linalg.solve(A, b)

    return w, A, b

def pi(cs, w):
    cs = np.array(cs)
    values = cs.dot(w)
    return np.argmax(values)

def eval_pi(w):
    rewards = []
    for k in TEST_PROCESSED.keys():
        sample = TEST_PROCESSED[k]            
        cs = sample[0]
        ys = sample[1]
        a = pi(cs, w)
        if not np.isnan(ys[a]):
            rewards.append(ys[a])

    return np.mean(rewards)

def get_result_fix_budget(method, n_eval_batch_samples, experiment_size):
    result_df = pd.DataFrame()
    values = []
    sample_cumu_rewards = []
    num_samples = []
    sample_cumu_costs = []
    budgets = []
    x_list = []
    y_list = []
    c_list = []
    decision_list = []
    
    train_keys = list(TRAIN_PROCESSED.keys())
    total_cost = 0  
    if 'S-P' in method:
        sp_alg = sample_offline(method, experiment_size)
    while len(x_list) <  experiment_size:
        print('method', method)
        print('len(x_list)', len(x_list))
        print(datetime.now())
        for n in range(len(train_keys)): 
            sample = TRAIN_PROCESSED[train_keys[n]]            
            cs = sample[0]
            ys = sample[1]
            costs = sample[2]
            if method == 'Random' or method == 'Min_cost':
                if method == 'Random':
                    decision = np.random.choice(np.arange(len(cs)))
            else:
                if 'S-P' in method:
                    if method == 'S-P':
                        decision = sp_alg.pi(cs)
                    elif method == 'S-P_cost' or method == 'S-P_cost_v2':
                        decision = sp_alg.pi(cs, costs)

            if not np.isnan(ys[decision]):
                x_list.append(cs[decision])
                y_list.append(ys[decision])
                cost = costs[decision]
                c_list.append(cost)
                total_cost += cost
                decision_list.append(decision)

                if len(x_list) % n_eval_batch_samples == 0 and len(x_list) >= n_eval_batch_samples:       
                    w, A, _, = learn(x_list, y_list, args.lam)
                    value = eval_pi(w)
                    values.append(value)          
                    sample_cumu_rewards.append(np.sum(y_list))                    
                    num_samples.append(len(x_list)) 
                    sample_cumu_costs.append(total_cost)                                         

    result_df = pd.DataFrame()
    result_df['average_reward'] = values
    result_df['sample_cumu_reward'] = sample_cumu_rewards
    result_df['num_samples'] = num_samples
    result_df['cumu_cost'] = sample_cumu_costs
    result_df['sample_policy'] = [method] * len(result_df)

    return result_df

def run(method):
    np.random.seed(args.random_seed)
    print('args.random_seed', args.random_seed)

    if method == 'S-P_cost':
        experiment_size = 500000
        n_eval_batch_samples = 500
    else:
        experiment_size = 50000
        n_eval_batch_samples = 50
    result = get_result_fix_budget(
        method = method, n_eval_batch_samples = n_eval_batch_samples, experiment_size = experiment_size
    )
    result['random_seed'] = [args.random_seed]*len(result)
    return result

result_rand = run('Random')
result_sp = run('S-P')
result_sp_cost = run('S-P_cost')
result_all = pd.concat([result_rand, result_sp, result_sp_cost])
result_all.to_csv(args.results_file_name + '_result_lam' + str(args.lam) + '_' + str(args.random_seed) + '_n' + args.train_file_name.split('_n')[-1].split('.pkl')[0] +'.csv', index=False)