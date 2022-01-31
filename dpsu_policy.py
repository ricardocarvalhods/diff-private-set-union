'''
Code obtained from (Gopi et. al 2020)'s repository: https://github.com/heyyjudes/differentially-private-set-union
'''

from enum import Enum
import numpy as np
import agm_example as agm
import scipy.stats
from collections import Counter, defaultdict
import operator
from tqdm import tqdm


class Noise(Enum):
    LAPLACE = 1
    GAUSSIAN = 2
    
def update_budget_dict(rho_dict, update):
    for key in rho_dict: 
        rho_dict[key] -= update 
    return rho_dict

def calculate_threshold_policy(noise, eps, delta, Delta_0):
    if noise == Noise.LAPLACE:
        l_param = 1 / eps
        F_l_rho = lambda t: 1 / t + (1 / eps) * np.lib.scimath.log(1 / (2 * (1 - (1 - delta) ** (1 / t))))
        l_rho = max([F_l_rho(t) for t in range(1, Delta_0 + 1)])
        return l_param, l_rho
    elif noise == Noise.GAUSSIAN:
        g_param = agm.calibrate_analytic_gaussian_mechanism(epsilon=eps, delta=delta / 2, GS=1, tol=1.e-12)
        F_g_rho = lambda t: 1 / np.lib.scimath.sqrt(t) + g_param * scipy.stats.norm.ppf((1 - delta / 2) ** (1 / t))
        g_rho = max([F_g_rho(t) for t in range(1, Delta_0 + 1)])
        return g_param, g_rho
    else:
        raise Exception("invalid noise and algorithm combination {} {}".format(algorithm, noise))

def update_budget_dict(rho_dict, update):
    '''
    Helper method to update uniformly update rho_dict
    :param rho_dict: input dictionary containing values to update
    :type rho_dict: dict
    :param update: amount to subtract each value in rho_dict by
    :return: update rho_dict
    '''
    for key in rho_dict: 
        rho_dict[key] -= update 
    return rho_dict

def run_policy(input_data, Delta_0, minha_dist, alpha, eps_ours, delta_ours):
    if minha_dist == 'GAUSSIAN':
        noise_dist = Noise.GAUSSIAN
    else:
        noise_dist = Noise.LAPLACE
        

    new_hist = defaultdict(float)

    ## Threshold and noise parameters
    g_param, g_rho = calculate_threshold_policy(noise_dist, eps_ours, delta_ours, Delta_0)
    Gamma = g_rho + alpha*g_param

    ## Added shuffling
    np.random.seed(1)
    np.random.shuffle(input_data)

    for all_grams in tqdm(input_data, position=0, leave=True):
        ## Sampling of Delta_0 items
        all_grams = list(set(all_grams))
        if len(all_grams) > Delta_0:
            selected_ngrams = np.random.choice(all_grams, size=Delta_0, replace=False).tolist()
        else:
            selected_ngrams = all_grams[:]

        ## Filter below cutoff
        selected_ngrams = [gram for gram in selected_ngrams if new_hist[gram] < Gamma]

        if len(selected_ngrams) > 0:                
            user_budget = 1

            if minha_dist == 'GAUSSIAN':
                # calculate normalization constant
                diff_arr = np.asarray([Gamma - new_hist[gram] for gram in selected_ngrams])                    
                Z = np.linalg.norm(diff_arr, ord=2)

                # add update to histogram proportional to distance to threshold
                for i, ngram in enumerate(selected_ngrams):
                    new_hist[ngram] += min(user_budget, Z)*diff_arr[i]/Z
            else:
                gap_dict = {}

                for w in selected_ngrams:
                    if new_hist[w] < Gamma:
                        gap_dict[w] = Gamma - new_hist[w]
                # sort rho dict
                sorted_gap_dict = sorted(gap_dict.items(), key=operator.itemgetter(1))
                sorted_gap_keys = [k for k, v in sorted_gap_dict]

                total_tokens = len(sorted_gap_keys)

                for i, w in enumerate(sorted_gap_keys):
                    cost = gap_dict[w]*(total_tokens-i)
                    if cost < user_budget:
                        for j in range(i, total_tokens):
                            add_gram = sorted_gap_keys[j]
                            new_hist[add_gram] += gap_dict[w]
                        # update remaining budget
                        user_budget -= cost
                        # update dictionary of values containing difference from gap
                        gap_dict = update_budget_dict(gap_dict, gap_dict[w])
                    else:
                        for j in range(i, total_tokens):
                            add_gram = sorted_gap_keys[j]
                            new_hist[add_gram] += user_budget/(total_tokens-i)
                        break

    ## Get items with noisy weights above threshold
    output_vocab = {}
    for key, val in new_hist.items():
        if noise_dist == Noise.LAPLACE:
            nval = val + np.random.laplace(0, g_param)
        else:
            nval = val + np.random.normal(0, g_param)
        if nval > g_rho:
            output_vocab[key] = val

    return len(output_vocab)
