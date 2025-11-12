import sys
from pathlib import Path
sys.path.append((str(Path(__file__).absolute().parent.parent)))
import pickle
import numpy as np
import tqdm
import random
import torch
from src.infer_policies import DiscretePolicyNetwork
from src.utils import MODELS, get_action_idx_hk, filter_annotations_by_subject_ids, restrict_annotation_count
from sklearn.linear_model import LinearRegression, LogisticRegression

# Seed
np.random.seed(42)

# Policies
state_dim=15
num_actions = 4
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
def calculate_gt_policy_value(target_dataset):
    return np.mean([reward_function(clin_feat) for clin_feat in target_dataset['clinical_features']])

def learn_rhat(b_data):
    factual_actions = b_data['actions']  # 0, 10, 20, 40
    clinical_features = b_data['clinical_features']
    XA = []
    R = []
    for context, action, clin_feat in zip(b_data['contexts'], factual_actions, clinical_features):
        # Factual sample
        context = np.nan_to_num(context, nan=0.0)
        XA.append(format_context_action(context, get_action_idx_hk(action)))
        R.append(reward_function(clin_feat))
    # print("Fitting Rhat, num_samples" + str(np.asarray(XA).shape))
    reg = LinearRegression().fit(np.asarray(XA), np.asarray(R))
    return reg

def learn_rhat_plus(b_data, c_annot, annot_budget):
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
        R.append(reward_function(clin_feat))

        # Counterfactual annotations
        if id in c_annot:
            for dosage in c_annot[id]:
                if isinstance(c_annot[id][dosage], list):
                    for annot in c_annot[id][dosage]:
                        if n_annot < annot_budget:
                            CA.append(format_context_action(context, get_action_idx_hk(dosage)))
                            G.append(reward_function(annot))
                            n_annot += 1
                else:
                    if n_annot < annot_budget:
                        CA.append(format_context_action(context, get_action_idx_hk(dosage)))
                        G.append(reward_function(c_annot[id][dosage]))
                        n_annot += 1

    if len(CA) > 0:
        # print("Fitting Rhatplus, num_samples" + str(np.vstack((XA, CA)).shape) + " size behavior_dataset: " + str(len(b_data['contexts'])))
        reg = LinearRegression().fit(np.vstack((XA, CA)), np.hstack((R, G)))
    else:
        # print("Fitting Rhatplus, num_samples" + str(np.array(XA).shape))
        reg = LinearRegression().fit(np.asarray(XA), np.asarray(R))
    return reg, n_annot

def reward_function(lab_value):
    '''Returns the scalar annotation as a function of the lab value'''

    # Option 1: indicator of reference range
    # annotation = 1 if (lab_value >=3.5 and lab_value <= 5) else 0

    # # Option 2: Gaussian Reward centered at mean of reference range
    # mean = 4.25  # Centered between 3.5 and 5
    # variance = 2.25
    # std_dev = np.sqrt(variance)
    #
    # # Gaussian density
    # density = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-((lab_value - mean) ** 2) / (2 * variance))
    #
    # # Normalize so max is 1
    # max_density = 1 / (np.sqrt(2 * np.pi * variance))
    # annotation = density / max_density

    # Option 3: step functions
    # if lab_value < 3:
    #     annotation = 0.4
    # elif lab_value >= 3 and lab_value < 3.5:
    #     annotation = 0.7
    # elif lab_value >= 3.5 and lab_value <= 5:
    #     annotation = 1
    # elif lab_value > 5 and lab_value <= 5.5:
    #     annotation = 0.7
    # elif lab_value > 5.5:
    #     annotation = 0.4

    # Option 4: Smooth Gaussian Decay + Plateau --> verified by Chloe
    annotation = 1
    # Left tail
    if lab_value < 3.5:
        sigma_left = 0.3
        annotation = np.exp(-0.5 * ((lab_value - 3.5) / sigma_left) ** 2)
    elif lab_value > 5.0:
        sigma_right = 0.3
        annotation = np.exp(-0.5 * ((lab_value - 4.5) / sigma_right) ** 2)
    return annotation

# Implement the methods for OPE
def standard_dm(b_data):
    R_hat = learn_rhat(b_data)
    DM = []
    for (context,  clin_feat) in zip(b_data['contexts'], b_data['clinical_features']):
        s_j = np.nan_to_num(np.asarray(context),
                            nan=0.0)  # We want all nans to be 0 now, but ideally we wouldn't have nans.
        dm_value = 0
        for a in range(num_actions):  # There are 4 possible actions
            x_a = format_context_action(s_j, a)
            dm_value += pi_e(normalize_sample(s_j, target_stat_name))[a].detach().numpy() * R_hat.predict(
                x_a.reshape(1, -1))
        DM.append(dm_value)
    return np.mean(DM)

def cdm(b_data, counterfactual_annotations, fake_annotations, num_annotations, aligned, model):
    # Filter annotations
    if model == 'fake':
        counterfactual_annotations = filter_annotations_by_subject_ids(fake_annotations, b_data['subject_ids'], model)
        # restricted_annotations, total_annot = restrict_annotation_count(fake_annotations, num_annotations)
    else:
        counterfactual_annotations = filter_annotations_by_subject_ids(counterfactual_annotations,
                                                                       b_data['subject_ids'], model)
        # restricted_annotations, _ = restrict_annotation_count(counterfactual_annotations, num_annotations)

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

def standard_is(behavior_dataset):
    IS = 0
    for (context, action, clin_feat) in zip(behavior_dataset['contexts'], behavior_dataset['actions'], behavior_dataset['clinical_features']):
        s_j = np.nan_to_num(np.asarray(context), nan=0.0) # We want all nans to be 0 now, but ideally we wouldn't have nans.
        a_j = action
        r_j = reward_function(clin_feat)
        a_j_idx = get_action_idx_hk(a_j)
        rho_e = pi_e(normalize_sample(s_j, target_stat_name))[a_j_idx]
        rho_b = pi_b(normalize_sample(s_j, behavior_stat_name))[a_j_idx]
        IS += (rho_e.detach().numpy()/rho_b.detach().numpy()).item() * r_j
    return IS/len(behavior_dataset['contexts'])# The number of samples
def dm_is(b_data):
    DM_IS = []
    R_hat = learn_rhat(b_data)

    for (context, action, clin_feat) in zip(b_data['contexts'], b_data['actions'], b_data['clinical_features']):
        s_j = np.nan_to_num(np.asarray(context), nan=0.0)  # We want all nans to be 0 now, but ideally we wouldn't have nans.
        a_j = action
        r_j = reward_function(clin_feat)
        a_j_idx = get_action_idx_hk(a_j)
        rho_e = pi_e(normalize_sample(s_j, target_stat_name))[a_j_idx] # This requires normalizing because our policies require normalizing.
        rho_b = pi_b(normalize_sample(s_j, behavior_stat_name))[a_j_idx]
        dm_value = 0
        for a in range(num_actions): # There are 4 possible actions
            x_a = format_context_action(s_j, a)
            dm_value += pi_e(normalize_sample(s_j, target_stat_name))[a].detach().numpy() * R_hat.predict(x_a.reshape(1, -1))

        is_component = (rho_e.detach().numpy()/rho_b.detach().numpy()).item() * (r_j - R_hat.predict(format_context_action(s_j, get_action_idx_hk(a_j)).reshape(1, -1)).item())
        dm_is_value = (dm_value + is_component)
        DM_IS.append(dm_is_value)
    return np.mean(DM_IS)

def cdm_is(b_data, counterfactual_annotations, fake_annotations, num_annotations, aligned, model):
    # Filter annotations
    if model == 'fake':
        counterfactual_annotations = filter_annotations_by_subject_ids(fake_annotations, b_data['subject_ids'], model)
        # restricted_annotations, total_annot = restrict_annotation_count(fake_annotations, num_annotations)
    else:
        counterfactual_annotations = filter_annotations_by_subject_ids(counterfactual_annotations, b_data['subject_ids'], model)
        # restricted_annotations, _ = restrict_annotation_count(counterfactual_annotations, num_annotations)  #

    C_DM_IS = []
    R_hat_plus, total_annot = learn_rhat_plus(b_data, counterfactual_annotations, annot_budget=num_annotations)
    for (context, action, clin_feat) in zip(b_data['contexts'], b_data['actions'], b_data['clinical_features']):
        s_j = np.nan_to_num(np.asarray(context), nan=0.0)  # We want all nans to be 0 now, but ideally we wouldn't have nans.
        a_j = action
        r_j = reward_function(clin_feat)
        a_j_idx = get_action_idx_hk(a_j)
        rho_e = (pi_e(normalize_sample(s_j, target_stat_name))[a_j_idx]).detach().numpy()
        rho_b = (pi_b(normalize_sample(s_j, behavior_stat_name))[a_j_idx]).detach().numpy()

        dm_value = 0
        for a in range(num_actions): # There are 4 possible actions # a is an index.
            x_a = format_context_action(s_j, a)
            dm_value += pi_e(normalize_sample(s_j, target_stat_name))[a].detach().numpy().item() * R_hat_plus.predict(x_a.reshape(1, -1))

        is_component = (rho_e/rho_b).item() * (r_j - R_hat_plus.predict(format_context_action(s_j, a_j_idx).reshape(1, -1)).item())
        cdm_is_value = (dm_value + is_component)
        C_DM_IS.append(cdm_is_value)
    return np.mean(C_DM_IS), total_annot

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

def construct_fake_annotations(real_annotations, behavior_dataset, low=2.0, high=8.0, upsample=False):
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

'''
Load data for OPE
'''

PI_Bs = ["../policies/hk_female_cb.pth", "../policies/nonrenal_cb.pth", "../policies/low_k_cb.pth"]
PI_Es = ["../policies/hk_male_cb.pth", "../policies/renal_cb.pth", "../policies/high_k_cb.pth"]
TARGET_STATS = ["../policies/hk_male_cb_", "../policies/renal_cb_", "../policies/high_k_cb_"]
BEHAVIOR_STATS = ["../policies/hk_female_cb_", "../policies/nonrenal_cb_", "../policies/low_k_cb_"]
TARGET_DATASETS = ["../notebooks/trajectories/male_cb_07172025.pkl", '../notebooks/trajectories/renal_cb.pkl', '../notebooks/trajectories/high_k_cb.pkl']
BEHAVIOR_DATSETS = ["../notebooks/trajectories/female_cb_07172025.pkl", '../notebooks/trajectories/nonrenal_cb.pkl', '../notebooks/trajectories/low_k_cb.pkl']
# RESULT_FNAMES = ["results/cb_gender_hk_08172025.pkl", "results/cb_comorbidity_hk_08172025.pkl", "results/cb_dosage_hk_08182025.pkl"]
RESULT_FNAMES = ["results/cb_gender_hk_both.pkl", 'results/cb_comorbidity_hk_both.pkl', 'results/cb_dosage_hk_both.pkl']
n_iter = 500  # The number of bootstrapped episodes

print(f"HYPERKALEMIA, n_iter={n_iter}")
for pi_b_fname, pi_e_fname, target_stat_name, behavior_stat_name, target_dataset_fname, behavior_dataset_fname, results_fname in zip(PI_Bs, PI_Es, TARGET_STATS, BEHAVIOR_STATS, TARGET_DATASETS, BEHAVIOR_DATSETS, RESULT_FNAMES):
    if pi_b_fname != "../policies/low_k_cb.pth": # Only dosage cohort
        continue
    target_dataset = pickle.load(open(target_dataset_fname, 'rb'))
    behavior_dataset = pickle.load(open(behavior_dataset_fname, 'rb'))

    pi_b = DiscretePolicyNetwork(state_dim=state_dim, num_actions=4)
    pi_b.load_state_dict(torch.load(pi_b_fname, weights_only=True))
    pi_b.eval()
    pi_e = DiscretePolicyNetwork(state_dim=state_dim, num_actions=4)
    pi_e.load_state_dict(torch.load(pi_e_fname, weights_only=True))
    pi_e.eval()

    gt_v_pie = calculate_gt_policy_value(target_dataset)

    counterfactual_annotations_fname = "../data/hk_data/all_annotations_rate=10_08082025.pkl" # Just adding new counterfactual annotations. (used to be 0702)
    counterfactual_annotations = pickle.load(open(counterfactual_annotations_fname, 'rb'))
    fake_annotations = construct_fake_annotations(counterfactual_annotations, behavior_dataset, low=2, high=8, upsample=True)

    aligned = False # Are the annotations aligned to the base distribution of reward in the behavior dataset?
    print("GT V_pie: " + str(gt_v_pie))

    results = {}
    alpha = 0.1
    N = len(behavior_dataset['subject_ids'])
    for m in ['o1', 'o3-mini', 'both_hk', 'average_hk']: # both is now
        print(str(m))
        n_annotations = [int(i*300) for i in range(8)] # These are additional annotations.
        used_annotations = []

        is_perf_mean_rmse = []
        is_perf_lb = []
        is_perf_ub = []

        dmis_perf_mean_rmse = []
        dmis_perf_lb = []
        dmis_perf_ub = []

        cdmis_perf_mean_rmse = []
        cdmis_perf_lb = []
        cdmis_perf_ub = []

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
            for _ in range(n_iter): # Each episode is a bootstrapping population.
                # Here, sample, with replacement, for the # of
                bootstrap_subject_ids = np.random.choice(behavior_dataset['subject_ids'], size=N, replace=True)
                b_dataset_bootstrap = filter_bootstrap_samples(behavior_dataset, bootstrap_subject_ids)
                # v_hat_is = standard_is(b_dataset_bootstrap)
                # v_hat_cdmis, used_annot = cdm_is(b_data=b_dataset_bootstrap, num_annotations=n_annot, model=m, counterfactual_annotations=counterfactual_annotations, fake_annotations=fake_annotations, aligned=aligned)
                v_hat_cdm, used_annot = cdm(b_data=b_dataset_bootstrap, num_annotations=n_annot, model=m, counterfactual_annotations=counterfactual_annotations, fake_annotations=fake_annotations, aligned=aligned)
                v_hat_dm = standard_dm(b_dataset_bootstrap)
                # v_hat_dmis = dm_is(b_dataset_bootstrap)

                used_annot_it.append(used_annot)

                # Calculate RMSE
                # rmse = np.sqrt((gt_v_pie - v_hat_is)**2)
                # is_rmses.append(rmse)

                # rmse = np.sqrt((gt_v_pie - v_hat_dmis)**2)
                # dmis_rmses.append(rmse)
                #
                # rmse = np.sqrt((gt_v_pie - v_hat_cdmis)**2)
                # cdmis_rmses.append(rmse)

                rmse = np.sqrt((gt_v_pie - v_hat_dm)**2)
                dm_rmses.append(rmse)

                rmse = np.sqrt((gt_v_pie - v_hat_cdm)**2)
                cdm_rmses.append(rmse)

            # is_perf_mean_rmse.append(np.mean(is_rmses))
            # lower = np.percentile(is_rmses, 100 * alpha / 2)
            # upper = np.percentile(is_rmses, 100 * (1 - alpha / 2))
            # is_perf_lb.append(lower)
            # is_perf_ub.append(upper)

            # dmis_perf_mean_rmse.append(np.mean(dmis_rmses))
            # lower = np.percentile(dmis_rmses, 100 * alpha / 2)
            # upper = np.percentile(dmis_rmses, 100 * (1 - alpha / 2))
            # dmis_perf_lb.append(lower)
            # dmis_perf_ub.append(upper)
            #
            # cdmis_perf_mean_rmse.append(np.mean(cdmis_rmses))
            # lower = np.percentile(cdmis_rmses, 100 * alpha / 2)
            # upper = np.percentile(cdmis_rmses, 100 * (1 - alpha / 2))
            # cdmis_perf_lb.append(lower)
            # cdmis_perf_ub.append(upper)

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

            results[m] = {'n_annotations': used_annotations, 'dmis_mean': dmis_perf_mean_rmse, 'dmis_lower': dmis_perf_lb,
                          'dmis_upper': dmis_perf_ub, 'cdmis_mean': cdmis_perf_mean_rmse, 'cdmis_lower':cdmis_perf_lb,
                          'cdmis_upper':cdmis_perf_ub, 'dm_mean':dm_perf_mean_rmse, 'dm_lower':dm_perf_lb,
                          'dm_upper':dm_perf_ub, 'cdm_mean':cdm_perf_mean_rmse, 'cdm_lower':cdm_perf_lb,
                          'cdm_upper':dm_perf_ub, 'dm_all':dm_all, 'cdm_all':cdm_all}
            pickle.dump(results, open(results_fname, 'wb'))


