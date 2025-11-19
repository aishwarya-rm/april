import sys
from pathlib import Path
sys.path.append((str(Path(__file__).absolute().parent.parent)))
import pickle
import numpy as np
import tqdm
import random
import torch
from src.infer_policies import DiscretePolicyNetwork
from src.utils import MODELS, get_action_idx_hk, filter_annotations_by_subject_ids
from sklearn.linear_model import LinearRegression

# Seed
np.random.seed(42)

# Policies
state_dim=15

def safe_nanmean(arr, default=0):
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return default
    return np.nanmean(arr)
def normalize_sample(state, stat_fname):
    mean = np.load(stat_fname + 'mean.npy')
    std = np.load(stat_fname + 'std.npy')
    return torch.Tensor((state - mean) / std)
def format_context_action(context, action_idx, num_actions=4, context_dim=15):
    x_a = np.zeros((num_actions, context_dim))
    x_a[action_idx, :] = context
    return x_a.flatten()
def calculate_gt_policy_value(target_dataset, task):
    return np.mean([reward_function(clin_feat, task) for clin_feat in target_dataset['clinical_features']])

def learn_rhat(b_data, task):
    factual_actions = b_data['actions']  # 0, 10, 20, 40
    clinical_features = b_data['clinical_features']
    XA = []
    R = []
    for context, action, clin_feat in zip(b_data['contexts'], factual_actions, clinical_features):
        # Factual sample
        context = np.nan_to_num(context, nan=0.0)
        XA.append(format_context_action(context, get_action_idx_hk(action)))
        R.append(reward_function(clin_feat, task))
    # print("Fitting Rhat, num_samples" + str(np.asarray(XA).shape))
    reg = LinearRegression().fit(np.asarray(XA), np.asarray(R))
    return reg

def learn_rhat_plus(b_data, c_annot, annot_budget, task):
    factual_actions = b_data['actions'] # 0, 10, 20, 40
    clinical_features = b_data['clinical_features']
    ids = b_data['subject_ids']
    XA = []
    R = []
    CA = []
    G = []
    n_annot = 0
    for context, action, clin_feat, id in zip(b_data['contexts'], factual_actions, clinical_features, ids):
        # Factual sample
        context = np.nan_to_num(context)
        XA.append(format_context_action(context, get_action_idx_hk(action)))
        R.append(reward_function(clin_feat, task))

        # Counterfactual annotations
        if id in c_annot:
            for dosage in c_annot[id]:
                if isinstance(c_annot[id][dosage], list):
                    for annot in c_annot[id][dosage]:
                        if n_annot < annot_budget:
                            CA.append(format_context_action(context, get_action_idx_hk(dosage)))
                            G.append(reward_function(annot, task))
                            n_annot += 1
                else:
                    if n_annot < annot_budget:
                        CA.append(format_context_action(context, get_action_idx_hk(dosage)))
                        G.append(reward_function(c_annot[id][dosage], task))
                        n_annot += 1

    if len(CA) > 0:
        reg = LinearRegression().fit(np.vstack((XA, CA)), np.hstack((R, G)))
    else:
        reg = LinearRegression().fit(np.asarray(XA), np.asarray(R))
    return reg, n_annot

def reward_function(lab_value, task):
    '''Returns the scalar annotation as a function of the lab value'''
    if task == 'potassium':
        annotation = 1
        # Left tail
        if lab_value < 3.5:
            sigma_left = 0.3
            annotation = np.exp(-0.5 * ((lab_value - 3.5) / sigma_left) ** 2)
        elif lab_value > 5.0:
            sigma_right = 0.3
            annotation = np.exp(-0.5 * ((lab_value - 4.5) / sigma_right) ** 2)
    elif task == 'sodium':
        annotation = 1
        # Left tail
        if lab_value < 135:
            sigma_left = 2.5
            annotation = np.exp(-0.5 * ((lab_value - 135) / sigma_left) ** 2)
        elif lab_value > 145:
            sigma_right = 2.5
            annotation = np.exp(-0.5 * ((lab_value - 145) / sigma_right) ** 2)
    return annotation

# Implement the methods for OPE
def standard_dm(b_data):
    R_hat = learn_rhat(b_data)
    DM = []
    for (context,  clin_feat) in zip(b_data['contexts'], b_data['clinical_features']):
        s_j = np.nan_to_num(np.asarray(context),
                            nan=0.0)  
        dm_value = 0
        for a in range(num_actions): 
            x_a = format_context_action(s_j, a)
            dm_value += pi_e(normalize_sample(s_j, target_stat_name))[a].detach().numpy() * R_hat.predict(
                x_a.reshape(1, -1))
        DM.append(dm_value)
    return np.mean(DM)

def cdm(b_data, counterfactual_annotations, fake_annotations, num_annotations, aligned, model):
    # Filter annotations
    if model == 'fake':
        counterfactual_annotations = filter_annotations_by_subject_ids(fake_annotations, b_data['subject_ids'], model)
    else:
        counterfactual_annotations = filter_annotations_by_subject_ids(counterfactual_annotations,
                                                                       b_data['subject_ids'], model)

    C_DM = []
    R_hat_plus, total_annot = learn_rhat_plus(b_data, counterfactual_annotations, annot_budget=num_annotations) # This allows us to have duplicate counterfactual annotations if we have the same subject id twice in our dataset.
    for (context, action, clin_feat) in zip(b_data['contexts'], b_data['actions'], b_data['clinical_features']):
        s_j = np.nan_to_num(np.asarray(context), nan=0.0)  # We want all nans to be 0 now, but ideally we wouldn't have nans.
        cdm_value = 0
        for a in range(num_actions): # There are 4 possible actions # a is an index.
            x_a = format_context_action(s_j, a)
            cdm_value += pi_e(normalize_sample(s_j, target_stat_name))[a].detach().numpy().item() * R_hat_plus.predict(x_a.reshape(1, -1))
        C_DM.append(cdm_value)
    return np.mean(C_DM), total_annot


def filter_bootstrap_samples(behavior_dataset, subject_ids):
    # TODO: Make this more efficient, we can treat these as big matrices.
    subset = {}
    subset['contexts'] = []
    subset['actions'] = []
    subset['clinical_features'] = []
    subset['subject_ids'] = []
    for pid in subject_ids:
        pid_idx = behavior_dataset['subject_ids'].index(pid)
        subset['contexts'].append(behavior_dataset['contexts'][pid_idx])
        subset['actions'].append(behavior_dataset['actions'][pid_idx])
        subset['clinical_features'].append(behavior_dataset['clinical_features'][pid_idx])
        subset['subject_ids'].append(pid.copy())
    assert len(subset['subject_ids']) == len(subject_ids), "The subset was not filtered properly."
    return subset

def construct_fake_annotations(real_annotations, behavior_dataset, low, high, upsample=False):
    if upsample:
        possible_dosages = [0, 10, 20, 40]
        fake_annotations = {}
        for i, (context, action, id) in enumerate(zip(behavior_dataset['contexts'], behavior_dataset['actions'], behavior_dataset['subject_ids'])):
            fake_annotations[id] = {}
            administered_dosage = possible_dosages[get_action_idx_hk(action)] # get the existing dosage of the drug
            for dosage in possible_dosages:
                if dosage != administered_dosage:
                    fake_annotations[id][dosage] = random.uniform(low, high)
    fake_annotations = {} # id --> dosage --> random number
    for id in real_annotations.keys():
        fake_annotations[id] = {}
        for dosage in real_annotations[id]['o1']: # Just picking the model that I anticipate has the most annotations.
            fake_annotations[id][dosage] = random.uniform(low, high)
    return fake_annotations

def run_ope(n_iter, counterfactual_annotations, pi_bs, pi_es, target_stats, behavior_stats, target_datasets, behavior_datasets, result_fnames, task):
    for pi_b_fname, pi_e_fname, target_stat_name, behavior_stat_name, target_dataset_fname, behavior_dataset_fname, results_fname in zip(pi_bs, pi_es, target_stats, behavior_stats, target_datasets, behavior_datasets, result_fnames):
        target_dataset = pickle.load(open(target_dataset_fname, 'rb'))
        behavior_dataset = pickle.load(open(behavior_dataset_fname, 'rb'))

        pi_b = DiscretePolicyNetwork(state_dim=state_dim, num_actions=4)
        pi_b.load_state_dict(torch.load(pi_b_fname, weights_only=True))
        pi_b.eval()
        pi_e = DiscretePolicyNetwork(state_dim=state_dim, num_actions=4)
        pi_e.load_state_dict(torch.load(pi_e_fname, weights_only=True))
        pi_e.eval()

        gt_v_pie = calculate_gt_policy_value(target_dataset)
        if task == 'potassium':
            fake_annotations = construct_fake_annotations(counterfactual_annotations, behavior_dataset, low=2, high=8, upsample=False)
        elif task == 'sodium':
            fake_annotations = construct_fake_annotations(counterfactual_annotations, behavior_dataset, low=100, high=175, upsample=False)

        print("GT V_pie: " + str(gt_v_pie))

        results = {}
        alpha = 0.1
        N = len(behavior_dataset['subject_ids'])
        for m in MODELS: # both is now
            print(str(m))
            n_annotations = [int(i*300) for i in range(8)] # These are additional annotations.
            used_annotations = []

            dm_perf_mean_rmse = []
            dm_perf_lb = []
            dm_perf_ub = []
            dm_all = []

            cdm_perf_mean_rmse = []
            cdm_perf_lb = []
            cdm_perf_ub = []
            cdm_all = []

            for _, n_annot in enumerate(tqdm.tqdm(n_annotations)):
                # is_rmses = []
                # dmis_rmses = []
                # cdmis_rmses = []
                dm_rmses = []
                cdm_rmses = []
                used_annot_it = []
                for _ in range(n_iter): 
                    bootstrap_subject_ids = np.random.choice(behavior_dataset['subject_ids'], size=N, replace=True)
                    b_dataset_bootstrap = filter_bootstrap_samples(behavior_dataset, bootstrap_subject_ids)
                    v_hat_cdm, used_annot = cdm(b_data=b_dataset_bootstrap, num_annotations=n_annot, model=m, counterfactual_annotations=counterfactual_annotations, fake_annotations=fake_annotations, aligned=aligned)
                    v_hat_dm = standard_dm(b_dataset_bootstrap)

                    used_annot_it.append(used_annot)


                    rmse = np.sqrt((gt_v_pie - v_hat_dm)**2)
                    dm_rmses.append(rmse)

                    rmse = np.sqrt((gt_v_pie - v_hat_cdm)**2)
                    cdm_rmses.append(rmse)

                dm_perf_mean_rmse.append(np.mean(dm_rmses))
                lower = np.percentile(dm_rmses, 100 * alpha/2)
                upper = np.percentile(dm_rmses, 100* (1-alpha/2))
                dm_perf_lb.append(lower)
                dm_perf_ub.append(upper)
                dm_all.append(dm_rmses)

                cdm_perf_mean_rmse.append(np.mean(cdm_rmses))
                lower = np.percentile(cdm_rmses, 100 * alpha / 2)
                upper = np.percentile(cdm_rmses, 100 * (1 - alpha / 2))
                cdm_perf_lb.append(lower)
                cdm_perf_ub.append(upper)
                cdm_all.append(cdm_rmses)

                assert len(used_annot_it) == n_iter, "Don't have the correct number of annotations per iteration."
                used_annotations.append(np.mean(used_annot_it))

                results[m] = {'n_annotations': used_annotations, 'dm_mean':dm_perf_mean_rmse, 'dm_lower':dm_perf_lb,
                            'dm_upper':dm_perf_ub, 'cdm_mean':cdm_perf_mean_rmse, 'cdm_lower':cdm_perf_lb,
                            'cdm_upper':dm_perf_ub, 'dm_all':dm_all, 'cdm_all':cdm_all}
                pickle.dump(results, open(results_fname, 'wb'))

if __name__ == 'main':
    '''
    Load data for OPE
    '''

    # OPE for potsasium datasets
    num_actions = 4
    PI_Bs = ["../policies/female_potassium.pth", "../policies/no_comorbidity_potassium.pth", "../policies/low_dosage_potassium.pth"]
    PI_Es = ["../policies/male_potassium.pth", "../policies/comorbidity_potassium.pth", "../policies/high_dosage_potassium.pth"]
    TARGET_STATS = ["../policies/male_potassium_", "../policies/comorbidity_potassium_", "../policies/high_dosage_potassium_"]
    BEHAVIOR_STATS = ["../policies/female_potassium_", "../policies/no_comorbidity_potassium_", "../policies/low_dosage_potassium_"]
    TARGET_DATASETS = ["../data/cohorts/male_potassium.pkl", '../data/cohorts/comorbidity_potassium.pkl', '../data/cohorts/high_dosage_potassium.pkl']
    BEHAVIOR_DATSETS = ["../data/cohorts/female_potassium.pkl", '../data/cohorts/no_comorbidity_potassium.pkl', '../data/cohorts/low_dosage_potassium.pkl']
    RESULT_FNAMES = ["results/gender_potassium.pkl", 'results/comorbidity_potassium.pkl', 'results/dosage_potassium.pkl']
    n_iter = 500  
    counterfactual_annotations = pickle.load(open("../data/potassium_annotations.pkl", 'rb'))

    run_ope(n_iter, counterfactual_annotations, PI_Bs, PI_Es, TARGET_STATS, BEHAVIOR_STATS, TARGET_DATASETS, BEHAVIOR_DATSETS, RESULT_FNAMES, task='sodium')
    
    # OPE for sodium datsets
    PI_Bs = ["../policies/female_sodium.pth", "../policies/no_comorbidity_sodium.pth", "../policies/low_dosage_sodium.pth"]
    PI_Es = ["../policies/male_sodium.pth", "../policies/comorbidity_sodium.pth", "../policies/high_dosage_sodium.pth"]
    TARGET_STATS = ["../policies/male_sodium_", "../policies/comorbidity_sodium_", "../policies/high_dosage_sodium_"]
    BEHAVIOR_STATS = ["../policies/female_sodium_", "../policies/no_comorbidity_sodium_", "../policies/low_dosage_sodium_"]
    TARGET_DATASETS = ["../data/cohorts/male_sodium.pkl", '../data/cohorts/comorbidity_sodium.pkl', '../data/cohorts/high_dosage_sodium.pkl']
    BEHAVIOR_DATSETS = ["../data/cohorts/female_sodium.pkl", '../data/cohorts/no_comorbidity_sodium.pkl', '../data/cohorts/low_dosage_sodium.pkl']
    RESULT_FNAMES = ["results/gender_sodium.pkl", 'results/comorbidity_sodium.pkl', 'results/dosage_sodium.pkl']
    n_iter = 500  
    counterfactual_annotations = pickle.load(open("../data/sodium_annotations.pkl", 'rb'))

    run_ope(n_iter, counterfactual_annotations, PI_Bs, PI_Es, TARGET_STATS, BEHAVIOR_STATS, TARGET_DATASETS, BEHAVIOR_DATSETS, RESULT_FNAMES, task='sodium')    


