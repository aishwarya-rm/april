'''
Things that are shared across all of the files
'''
import numpy as np

url_o1 = ""
url_gpt4 = ""
url_gpt4o_mini = ""
url_o3_mini = ""
url_gemini = ""
url_claude = ""

MODELS = ['o1', 'gpt-4o-mini', 'o3-mini', 'gemini', 'claude-3.7']

MODEL_TO_URL = {'o1':url_o1, 'gpt-4':url_gpt4, 'gpt-4o-mini':url_gpt4o_mini, 'gemini':url_gemini, 'claude-3.7':url_claude, 'o3-mini':url_o3_mini}
HEADERS = {
        'Ocp-Apim-Subscription-Key': "",
        'Content-Type': 'application/json'
    }
POTASSIUM_ACTIONS = [0, 10, 20, 40]
SODIUM_ACTIONS = [0, 100, 200, 300, 400, 500]

def restrict_annotation_count(counterfactual_annotations, num_annotations):
    new_annotations = {}
    total_annot = 0
    for i, subject_id in enumerate(list(counterfactual_annotations.keys())):
        annotations = counterfactual_annotations[subject_id].keys()
        if (total_annot + len(annotations) < num_annotations) and (len(annotations) > 0):
            # new_annotations[subject_id] = counterfactual_annotations[subject_id].copy()
            new_annotations[subject_id] = {}
            for d in annotations:
                new_annotations[subject_id][d] = counterfactual_annotations[subject_id][d]
            total_annot += len(annotations)
    return new_annotations, total_annot

def merge_value_dicts(d1, d2):
    merged = {}
    all_keys = set(d1) | set(d2)
    for key in all_keys:
        if key in d1 and key in d2:
            merged[key] = [d1[key], d2[key]]
        elif key in d1:
            merged[key] = d1[key]
        else:
            merged[key] = d2[key]
    return merged
def average_value_dicts(d1, d2):
    averaged = {}
    all_keys = set(d1) | set(d2) # all possible dosages
    for key in all_keys: # all dosages
        if key in d1 and key in d2: # we need to average the values.
            if np.isnan(np.mean([d1[key], d2[key]])) or np.isinf(np.mean([d1[key], d2[key]])) or np.mean([d1[key], d2[key]]) > 1000:
                print("Average was instable. ")
                continue
            else:
                averaged[key] = np.mean([d1[key], d2[key]]).item()
        elif key in d1:
            averaged[key] = d1[key]
        else:
            averaged[key] = d2[key]
    return averaged
def filter_annotations_by_subject_ids(counterfactual_annotations, subject_ids, model):
    filtered_annotations = {}
    for id in subject_ids:
        if id in counterfactual_annotations.keys():
            if model == 'fake': # then the counterfactual annotations are fake
                filtered_annotations[id] = counterfactual_annotations[id]
            elif model == 'both_hk': # use both o1 and o3-mini
                if 'o1' in counterfactual_annotations[id].keys():
                    if 'o3-mini' in counterfactual_annotations[id].keys(): # both sets of annotations are available
                        filtered_annotations[id] = merge_value_dicts(counterfactual_annotations[id]['o1'], counterfactual_annotations[id]['o3-mini'])
                    else: # claude isn't available
                        filtered_annotations[id] = counterfactual_annotations[id]['o1']
                elif 'o3-mini' in counterfactual_annotations[id].keys(): # o1 not available
                    filtered_annotations[id] = counterfactual_annotations[id]['o3-mini']
            elif model == 'average_hk':
                if 'o1' in counterfactual_annotations[id].keys():
                    if 'o3-mini' in counterfactual_annotations[id].keys(): # both sets of annotations are available
                        filtered_annotations[id] = average_value_dicts(counterfactual_annotations[id]['o1'], counterfactual_annotations[id]['o3-mini'])
                    else: # claude isn't available
                        filtered_annotations[id] = counterfactual_annotations[id]['o1']
                elif 'o3-mini' in counterfactual_annotations[id].keys(): # o1 not available
                    filtered_annotations[id] = counterfactual_annotations[id]['o3-mini']
            elif model == 'both_hn': # use both gemini and o3-mini
                if 'gemini' in counterfactual_annotations[id].keys():
                    if 'o3-mini' in counterfactual_annotations[id].keys():  # both sets of annotations are available
                        filtered_annotations[id] = merge_value_dicts(counterfactual_annotations[id]['gemini'],
                                                                     counterfactual_annotations[id]['o3-mini'])
                    else:
                        filtered_annotations[id] = counterfactual_annotations[id]['gemini']
                elif 'o3-mini' in counterfactual_annotations[id].keys():  # o1 not available
                    filtered_annotations[id] = counterfactual_annotations[id]['o3-mini']
            elif model == 'average_hn':
                if 'gemini' in counterfactual_annotations[id].keys():
                    if 'o3-mini' in counterfactual_annotations[id].keys():  # both sets of annotations are available
                        filtered_annotations[id] = average_value_dicts(counterfactual_annotations[id]['gemini'],
                                                                       counterfactual_annotations[id]['o3-mini'])
                    else:  # claude isn't available
                        filtered_annotations[id] = counterfactual_annotations[id]['gemini']
                elif 'o3-mini' in counterfactual_annotations[id].keys():
                    filtered_annotations[id] = counterfactual_annotations[id]['o3-mini']
            else:
                if model in counterfactual_annotations[id].keys():
                    filtered_annotations[id] = counterfactual_annotations[id][model]
    return filtered_annotations
def find_sandwich(kcl_meds, serum_k_labs):
    for i, row_med in kcl_meds.iterrows():
        med_starttime = row_med['starttime']
        med_endtime = row_med['endtime']
        med_dosage = row_med['amount']
        med_unit = row_med['amountuom']

        # Lab before
        labs_before = []
        lab_times_before = []
        for index, row in serum_k_labs.iterrows():
            if row['charttime'] < med_starttime:
                labs_before.append(row['valuenum'])
                lab_times_before.append(row['charttime'])

        # Lab after
        labs_after = []
        lab_times_after = []
        for index, row in serum_k_labs.iterrows():
            if row['charttime'] > med_endtime:
                labs_after.append(row['valuenum'])
                lab_times_after.append(row['charttime'])

        if len(labs_before) >= 1 and len(labs_after) >= 1:  # This constitutes a valid sandwich
            lab_before = labs_before[-1]
            lab_time_before = lab_times_before[-1]  # The last lab before administration of potassium

            lab_after = labs_after[0]
            lab_time_after = lab_times_after[0]  # The first lab after administration of potassium

            assert lab_time_after >= med_endtime and lab_time_before <= med_starttime

            return med_dosage, med_starttime, med_endtime, lab_before, lab_time_before, lab_after, lab_time_after

    return None, None, None, None, None, None, None

def get_action_idx_hk(raw_action):
    act = round(raw_action, -1)
    if act == 0:
        idx=0
    elif act == 10:
        idx=1
    elif act == 20:
        idx=2
    elif act >= 30:  # Some actions are 30, but most are 40 or over.
        idx=3
    else:
        raise Exception()
    return idx
def get_action_idx_hn(raw_action):
    act = round(raw_action, -2)
    if act == 0:
       idx= 0
    elif act == 100:
        idx=1
    elif act == 200:
        idx=2
    elif act == 300:
        idx=3
    elif act == 400:
        idx=4
    elif act >= 500:
        idx=5
    else:
        raise Exception()
    return idx
