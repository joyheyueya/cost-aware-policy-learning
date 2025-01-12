import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
pandas2ri.activate()
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
from datetime import datetime
import argparse
import warnings
import os
from copy import deepcopy
import random
import torch
torch.set_default_dtype(torch.float64)
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--max_budget', default=10000, type=int)
parser.add_argument('--n_eval_batch_budget', default=100, type=int)
parser.add_argument('--eval_initial_budget', default=100, type=int)
parser.add_argument('--experiment_size', default=1500, type=int)
parser.add_argument('--results_file_name', default='results_rideshare/rideshare', type=str)
args = parser.parse_args()

if not os.path.exists(args.results_file_name.split('/')[0]):
    os.makedirs(args.results_file_name.split('/')[0])
predictors = ['est_intercept_coef', 'est_targetgroup_coef',
                  'est_is_felony_coef', 'est_is_male_coef', 'est_age_coef',
                  'est_log_distance_coef', 'est_fta_hist_coef', 'est_appt_hist_inv_coef',
                  'est_rideshare_coef', 'est_fta_hist_appt_hist_inv_coef', 'est_log_distance_transit_coef']
    
def get_decision(inferred_df, method):
    inferred_df['is_warmup'] = [False]*len(inferred_df)
    action_names = ['control','transit','rideshare']
    decision_list = []
    control_decision_list = []
    rideshare_decision_list = []
    transit_decision_list = []
    observed_appear_outcome_list = []
    observed_true_appear_prob_list = []
    if method == 'IG' or method == 'IG_cost':
        inferred_df = inferred_df.reset_index()
        M = len(inferred_df)
        inferred_df_prob = inferred_df[['control_inferred_appear_prob', 'transit_inferred_appear_prob', 'rideshare_inferred_appear_prob']]
        max_df = inferred_df_prob.idxmax(axis=1)
        
        Theta = {}
        for i in max_df.index:
            a = max_df.loc[i].split('_')[0]
            if a == 'control':
                if 0 in Theta:
                    Theta[0].append(i)
                else:
                    Theta[0] = [i]
            elif a == 'transit':
                if 1 in Theta:
                    Theta[1].append(i)
                else:
                    Theta[1] = [i]        
            elif a == 'rideshare':
                if 2 in Theta:
                    Theta[2].append(i)
                else:
                    Theta[2] = [i] 
        
        p_astar = {}
        for key in Theta.keys():
            p_astar[key] = len(Theta[key])/M

        y1_prob = inferred_df_prob.sum().values
        y0_prob = 1*len(inferred_df) - y1_prob
        p_a_y = np.array([y0_prob, y1_prob]).transpose()/M                    
        
        p_astar_a_y = deepcopy(p_astar)
        for key in p_astar_a_y.keys():
            p_astar_a_y[key] = np.zeros((3,2))
            y1_prob = inferred_df_prob.loc[Theta[key]].sum().values
            y0_prob = 1*len(inferred_df_prob.loc[Theta[key]]) - y1_prob
            p_astar_a_y[key] = np.array([y0_prob, y1_prob]).transpose()/M  
        
        g_a = [0,0,0]
        for i in range(3):
            for astar in p_astar_a_y.keys():
                for y in [0,1]:
                    if p_astar_a_y[astar][i][y] != 0.0:
                        g_a[i] += p_astar_a_y[astar][i][y]*np.log(p_astar_a_y[astar][i][y] / (p_astar[astar] * p_a_y[i][y]))
        inferred_df = inferred_df.head(1)
        for n in range(len(action_names)):
            inferred_df[action_names[n] + '_info'] = g_a[n]
            if method == 'IG_cost':
                inferred_df[action_names[n] + '_info_by_cost'] = inferred_df[action_names[n] + '_info'] / inferred_df[action_names[n] + '_cost']
        if method == 'IG':
            decision = inferred_df[['control_info', 'transit_info', 'rideshare_info']].idxmax(axis=1).values[0].split('_')[0]
        elif method == 'IG_cost':
            decision = inferred_df[['control_info_by_cost', 'transit_info_by_cost', 'rideshare_info_by_cost']].idxmax(axis=1).values[0].split('_')[0]
        decision_list.append(decision)
        observed_appear_outcome_list.append(inferred_df[decision + '_true_appear_outcome'].values[0])
        observed_true_appear_prob_list.append(inferred_df[decision + '_true_appear_prob'].values[0])
        if decision == 'control':
            control_decision_list.append(1)
            rideshare_decision_list.append(0)
            transit_decision_list.append(0)
        elif decision == 'rideshare':
            control_decision_list.append(0)
            rideshare_decision_list.append(1)
            transit_decision_list.append(0)
        elif decision == 'transit':
            control_decision_list.append(0)
            rideshare_decision_list.append(0)
            transit_decision_list.append(1)        
    else:
        for p in range(len(inferred_df)):
            if method == 'Random':
                decision = np.random.choice(action_names, 1)[0]
            elif method == 'Thompson':
                inferred_df_p = inferred_df.iloc[[p]][['control_inferred_appear_prob', 'transit_inferred_appear_prob', 'rideshare_inferred_appear_prob']]
                decision = inferred_df_p.idxmax(axis=1).values[0].split('_')[0]
            elif method == 'Thompson_cost':
                for action in action_names:
                    inferred_df[action + '_inferred_appear_prob_by_cost'] = inferred_df[action + '_inferred_appear_prob'] / inferred_df[action + '_cost']
                inferred_df_p = inferred_df.iloc[[p]][['control_inferred_appear_prob_by_cost', 'transit_inferred_appear_prob_by_cost', 'rideshare_inferred_appear_prob_by_cost']]
                decision = inferred_df_p.idxmax(axis=1).values[0].split('_')[0]  
            elif method == 'Oracle':
                inferred_df_p = inferred_df.iloc[[p]][['control_true_appear_prob', 'transit_true_appear_prob', 'rideshare_true_appear_prob']]
                decision = inferred_df_p.idxmax(axis=1).values[0].split('_')[0]              
            decision_list.append(decision)
            observed_appear_outcome_list.append(inferred_df.iloc[[p]][decision + '_true_appear_outcome'].values[0])
            observed_true_appear_prob_list.append(inferred_df.iloc[[p]][decision + '_true_appear_prob'].values[0])
            if decision == 'control':
                control_decision_list.append(1)
                rideshare_decision_list.append(0)
                transit_decision_list.append(0)
            elif decision == 'rideshare':
                control_decision_list.append(0)
                rideshare_decision_list.append(1)
                transit_decision_list.append(0)
            elif decision == 'transit':
                control_decision_list.append(0)
                rideshare_decision_list.append(0)
                transit_decision_list.append(1)

    inferred_df['decision'] = decision_list
    inferred_df['control_decision'] = control_decision_list
    inferred_df['rideshare_decision'] = rideshare_decision_list
    inferred_df['transit_decision'] = transit_decision_list
    inferred_df['observed_appear_outcome'] = observed_appear_outcome_list
    inferred_df['observed_true_appear_prob'] = observed_true_appear_prob_list
    return inferred_df

def get_result_fix_budget(method, max_budget, n_eval_batch_budget, eval_initial_budget, pop_df, test_pop, warmup_pop_df, experiment_size, calc_inferred_appear_prob_thompson, calc_inferred_appear_prob_mle, get_mle_coef):
    result_df = pd.DataFrame()
    train_pop_df = pd.DataFrame()
    train_pop_df = pd.concat([train_pop_df, warmup_pop_df]) 
    with localconverter(robjects.default_converter + pandas2ri.converter):
        train_pop = robjects.conversion.py2rpy(train_pop_df) 
    values = []
    sample_cumu_rewards = []
    num_samples = []
    sample_cumu_costs = []
    budgets = []
    sample_list = np.random.choice(range(experiment_size), experiment_size, replace=False)
    total_cost = 0                                
    for n in range(experiment_size):
        if n % 100 == 0:
            print(str(n), datetime.now())
        sample_pop_df = pop_df.iloc[[sample_list[n]]]
        with localconverter(robjects.default_converter + pandas2ri.converter): 
            sample_pop = robjects.conversion.py2rpy(sample_pop_df)  

        if method == 'Random':   
            sample_pop_inferred_df = get_decision(sample_pop_df, method)                
        elif method == 'Thompson' or method == 'Thompson_cost':
            sample_pop_inferred = calc_inferred_appear_prob_thompson(train_pop, sample_pop, 1)
            with localconverter(robjects.default_converter + pandas2ri.converter):
                sample_pop_inferred_df = robjects.conversion.rpy2py(sample_pop_inferred)                 
            sample_pop_inferred_df = get_decision(sample_pop_inferred_df, method)
        elif method == 'IG' or method == 'IG_cost':
            sample_pop_inferred = calc_inferred_appear_prob_thompson(train_pop, sample_pop, 10)
            with localconverter(robjects.default_converter + pandas2ri.converter):
                sample_pop_inferred_df = robjects.conversion.rpy2py(sample_pop_inferred)    
            sample_pop_inferred_df = get_decision(sample_pop_inferred_df, method)

        decision = sample_pop_inferred_df['decision'].values[0]
        cost = sample_pop_inferred_df[decision + '_cost'].values[0]

        total_cost += cost
        if 'index' in sample_pop_inferred_df.columns:
            sample_pop_inferred_df = sample_pop_inferred_df.drop(['index'], axis=1)
        train_pop_df = pd.concat([train_pop_df, sample_pop_inferred_df])
        with localconverter(robjects.default_converter + pandas2ri.converter):
            train_pop = robjects.conversion.py2rpy(train_pop_df)

        mle_coef = get_mle_coef(train_pop)
        with localconverter(robjects.default_converter + pandas2ri.converter):
            mle_coef_df = robjects.conversion.rpy2py(mle_coef)
        for predictor in predictors:
            if predictor not in mle_coef_df.columns:
                mle_coef_df[predictor] = 0.0
        with localconverter(robjects.default_converter + pandas2ri.converter):
            mle_coef = robjects.conversion.py2rpy(mle_coef_df)
            test_pop_inferred = calc_inferred_appear_prob_mle(mle_coef, test_pop)    
        with localconverter(robjects.default_converter + pandas2ri.converter):
            test_pop_inferred_df = robjects.conversion.rpy2py(test_pop_inferred)
        # Thompson sampling uses greedy decision
        test_pop_inferred_df = get_decision(test_pop_inferred_df, 'Thompson')                        
        values.append(test_pop_inferred_df['observed_true_appear_prob'].mean())            
        sample_cumu_rewards.append(train_pop_df['observed_appear_outcome'].sum())                    
        num_samples.append(len(train_pop_df) - len(warmup_pop_df)) 
        sample_cumu_costs.append(total_cost)
    result_df = pd.DataFrame()
    result_df['average_reward'] = values
    result_df['sample_cumu_reward'] = sample_cumu_rewards
    result_df['num_samples'] = num_samples
    result_df['cumu_cost'] = sample_cumu_costs
    result_df['sample_policy'] = [method] * len(result_df)

    return result_df, train_pop_df, test_pop_inferred_df

def get_GP_UCB_train_policy_value(gp_train_x_single, sample_pop, calc_inferred_appear_prob_mle):
    coef_df = pd.DataFrame({col: [val] for col, val in zip(['model_ix'] + predictors, [1] + gp_train_x_single.squeeze().tolist())})
    with localconverter(robjects.default_converter + pandas2ri.converter):
        coef = robjects.conversion.py2rpy(coef_df)
        sample_pop_inferred = calc_inferred_appear_prob_mle(coef, sample_pop)    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        sample_pop_inferred_df = robjects.conversion.rpy2py(sample_pop_inferred)
    # Thompson sampling uses greedy decision
    sample_pop_inferred_df = get_decision(sample_pop_inferred_df, 'Thompson')   
    return torch.tensor(sample_pop_inferred_df['observed_true_appear_prob'].mean()).unsqueeze(-1), sample_pop_inferred_df
    
def get_result_GP_UCB(method, max_budget, n_eval_batch_budget, eval_initial_budget, pop_df, test_pop, warmup_pop_df, experiment_size, calc_inferred_appear_prob_thompson, calc_inferred_appear_prob_mle, get_mle_coef):
    TRAIN_POLICY_EVAL_SIZE = 5
    result_df = pd.DataFrame()
    gp_train_x = []
    gp_train_y  = []
    optimal_theta_list = []
    values = []
    sample_cumu_rewards = []
    num_samples = []
    sample_cumu_costs = []
    budgets = []
    sample_list = np.random.choice(range(experiment_size), experiment_size, replace=False)
    total_cost = 0
    
    #initialize GP_UCB
    train_pop_df = pd.DataFrame()
    train_pop_df = pd.concat([train_pop_df, warmup_pop_df]) 
    with localconverter(robjects.default_converter + pandas2ri.converter):
        train_pop = robjects.conversion.py2rpy(train_pop_df) 
  
    # use 10 data points to evaluate policy paramters proposed by GP_UCB at each step
    for n in range(0, experiment_size, TRAIN_POLICY_EVAL_SIZE): 
        sample_pop_df = pop_df.iloc[sample_list[n:n+TRAIN_POLICY_EVAL_SIZE]]
        with localconverter(robjects.default_converter + pandas2ri.converter):
            sample_pop = robjects.conversion.py2rpy(sample_pop_df)    
        
        if len(gp_train_y) == 0:
            mle_coef = get_mle_coef(train_pop)
            with localconverter(robjects.default_converter + pandas2ri.converter):
                mle_coef_df = robjects.conversion.rpy2py(mle_coef)
            for predictor in predictors:
                if predictor not in mle_coef_df.columns:
                    mle_coef_df[predictor] = 0.0
            gp_train_x = torch.tensor(mle_coef_df[predictors].values)
            gp_train_y, sample_pop_inferred_df = get_GP_UCB_train_policy_value(gp_train_x, sample_pop, calc_inferred_appear_prob_mle)
        else:
            # Define and fit the GP model
            gp = SingleTaskGP(gp_train_x, gp_train_y.unsqueeze(-1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)
            # Define the acquisition function
            UCB = UpperConfidenceBound(gp, beta=0.1)  # Beta is a hyperparameter

            new_x, ucb_value = optimize_acqf(
                UCB, 
                bounds=torch.stack([-5*torch.ones(len(predictors)), 5*torch.ones(len(predictors))]), # this assumes x in [-10,10]^D
                q=1, 
                num_restarts=5, 
                raw_samples=50,
            )

            # Evaluate the objective function
            new_y, sample_pop_inferred_df = get_GP_UCB_train_policy_value(new_x, sample_pop, calc_inferred_appear_prob_mle)

            # Update training points
            gp_train_x = torch.cat([gp_train_x, new_x])
            gp_train_y = torch.cat([gp_train_y, new_y])
        
        cost = 0
        for s in range(len(sample_pop_inferred_df)):
            decision = sample_pop_inferred_df.iloc[s]['decision']
            cost += sample_pop_inferred_df.iloc[s][decision + '_cost']
            
        total_cost += cost
        train_pop_df = pd.concat([train_pop_df, sample_pop_inferred_df])

        optimal_theta = gp_train_x[gp_train_y.argmax()]  
        optimal_theta_list.append(optimal_theta)
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            test_pop_df = robjects.conversion.rpy2py(test_pop)
        value, test_pop_inferred_df = get_GP_UCB_train_policy_value(optimal_theta, test_pop, calc_inferred_appear_prob_mle)
        print(n)
        print("value", value) 
        print(datetime.now())
        values.append(value.squeeze().item())            
        sample_cumu_rewards.append(train_pop_df['observed_appear_outcome'].sum())                    
        num_samples.append(len(train_pop_df) - len(warmup_pop_df)) 
        sample_cumu_costs.append(total_cost)                                         
            
    result_df = pd.DataFrame()
    result_df['average_reward'] = values
    result_df['sample_cumu_reward'] = sample_cumu_rewards
    result_df['num_samples'] = num_samples
    result_df['cumu_cost'] = sample_cumu_costs
    result_df['sample_policy'] = [method] * len(result_df)
    result_df['optimal_theta'] = optimal_theta_list

    return result_df, train_pop_df, test_pop_inferred_df

def run(method):
    # Source the R scripts
    print('method', method)
    r_lang = robjects.r
    r_lang['source']('court_simulation_code/population_withcosts.R')
    r_lang['source']('court_simulation_code/inference_withcosts.R')
    r_lang['source']('court_simulation_code/act_withcosts.R')
    r_lang['source']('court_simulation_code/learn_withcosts_small.R')
    params = r_lang.list(param_rideshare_coef = 4,
                       param_transit_log_distance_coef = -0.75,
                       param_rideshare_distance_price_permi = 10,
                       param_transit_price_constant = 7.5,
                       param_max_distance = 20)

    robjects.r('set.seed(' + str(args.random_seed) + ')')
    np.random.seed(args.random_seed)
    print('args.random_seed', args.random_seed)

    generate_samples = robjects.globalenv['generate_samples']
    get_warmup_pop = robjects.globalenv['get_warmup_pop']
    calc_inferred_appear_prob_thompson = robjects.globalenv['calc_inferred_appear_prob_thompson']
    calc_inferred_appear_prob_mle = robjects.globalenv['calc_inferred_appear_prob_mle']
    get_mle_coef = robjects.globalenv['get_mle_coef']

    pop = generate_samples(params, args.experiment_size) 
    test_pop = generate_samples(params, args.experiment_size) 
    warmup_pop = get_warmup_pop(sim_params = params)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        pop_df = robjects.conversion.rpy2py(pop)
        warmup_pop_df = robjects.conversion.rpy2py(warmup_pop)
        
    warmup_pop_df = warmup_pop_df.drop(['control_policy_treat_prob',
           'transit_policy_treat_prob', 'rideshare_policy_treat_prob','observed_reward',
           'observed_spending'], axis=1)
    
    if method == 'GP-UCB':
        result, train, test_final = get_result_GP_UCB(method=method, max_budget=args.max_budget, 
                                                      n_eval_batch_budget=args.n_eval_batch_budget, 
                                                      eval_initial_budget=args.eval_initial_budget, 
                                                      pop_df=pop_df, test_pop=test_pop, warmup_pop_df=warmup_pop_df, 
                                                      experiment_size = args.experiment_size, 
                                                      calc_inferred_appear_prob_thompson=calc_inferred_appear_prob_thompson,
                                                      calc_inferred_appear_prob_mle=calc_inferred_appear_prob_mle,
                                                      get_mle_coef=get_mle_coef
                                                     )
    else:
        result, train, test_final = get_result_fix_budget(method=method, max_budget=args.max_budget, 
                                                      n_eval_batch_budget=args.n_eval_batch_budget, 
                                                      eval_initial_budget=args.eval_initial_budget, 
                                                      pop_df=pop_df, test_pop=test_pop, warmup_pop_df=warmup_pop_df, 
                                                      experiment_size = args.experiment_size, 
                                                      calc_inferred_appear_prob_thompson=calc_inferred_appear_prob_thompson,
                                                      calc_inferred_appear_prob_mle=calc_inferred_appear_prob_mle,
                                                      get_mle_coef=get_mle_coef
                                                     )
    result['random_seed'] = [args.random_seed]*len(result)
    train['random_seed'] = [args.random_seed]*len(train)
    test_final['random_seed'] = [args.random_seed]*len(test_final)
    train.to_csv(args.results_file_name + '_train_' + method + '_' + str(args.random_seed) + '.csv', index=False)
    test_final.to_csv(args.results_file_name + '_test_final_' + method + '_' + str(args.random_seed) + '.csv', index=False)
    return result

result_gpucb = run('GP-UCB')
result_rand = run('Random')
result_ts = run('Thompson')
result_ig = run('IG')
result_ig_cost = run('IG_cost')
result_all = pd.concat([result_rand, result_ts, result_ig, result_ig_cost, result_gpucb])
result_all.to_csv(args.results_file_name + '_result_' + str(args.random_seed) + '.csv', index=False)