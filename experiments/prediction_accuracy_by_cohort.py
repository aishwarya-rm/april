import sys
from pathlib import Path
sys.path.append((str(Path(__file__).absolute().parent.parent)))
from src.llm_prompting import prompt_llm, read_lab_prediction, read_lab_summary
import pickle
import tqdm
import time
import numpy as np
from src.utils import MODELS

# For Hyponatremia Task
# Three cohorts.
low_na_ids = pickle.load(open('../data/hn_data/low_na_ids.pkl', 'rb'))
female_ids = pickle.load(open('../data/hn_data/female_ids.pkl', 'rb'))
no_cirrhosis_ids = pickle.load(open('../data/hn_data/no_cirrhosis_ids.pkl', 'rb'))

prediction_prompts = pickle.load(open('../data/hn_data/prediction_prompts_07312025.pkl', 'rb')) # EHR --> Lab Value
num_tries = 2
saved_fname = '../data/hn_data/gt_lab_predictions_08042024.pkl'
dump_fname = '../data/hn_data/gt_lab_predictions_08102025.pkl'

lab_predictions = pickle.load(open(saved_fname, 'rb'))
for _, id in enumerate(tqdm.tqdm(prediction_prompts.keys())):
    lab_pred_prompt = prediction_prompts[id]['prediction_prompt']
    if (id not in low_na_ids) and (id not in female_ids) and (id not in no_cirrhosis_ids):
        continue
    if id not in lab_predictions.keys():
        lab_predictions[id] = {}
    for model in MODELS:
        if model not in lab_predictions[id].keys():
            curr_tries = 0 # Predict the lab value directly from the EHR
            success = False
            while not success and curr_tries < num_tries:
                try:
                    curr_tries += 1
                    raw_prediction = prompt_llm(prediction_prompts[id]['prediction_prompt'], model=model)
                    # Read in the prediction
                    lab_after = prediction_prompts[id]['gt_lab_after']
                    pred = read_lab_prediction(raw_prediction, model)
                    print("ID: " + str(id) + " Model: " + str(model) + " GT Lab value: " + str(lab_after) + " Predicted Lab value: " + str(pred))
                    success = True
                    lab_predictions[id][model] = pred
                except Exception as e:
                    wait = np.random.uniform(10, 50)
                    print(f"Error ({e}), model={model}, try={curr_tries}, Waiting {wait:.1f}s before retrying...")
                    time.sleep(wait)
        pickle.dump(lab_predictions, open(dump_fname, 'wb'))


# For Potassium
# low_k_cohort = pickle.load(open('../notebooks/trajectories/hk_low_dosage.pkl', 'rb'))
# nonrenal_cohort = pickle.load(open('../notebooks/trajectories/hk_nonrenal.pkl', 'rb'))
# female_cohort = pickle.load(open('../notebooks/trajectories/hk_female_ids.pkl', 'rb'))
# 
# prediction_prompts = pickle.load(open('../data/hk_data/prediction_prompts_06052025.pkl', 'rb')) # EHR --> Lab Value
# num_tries = 2
# saved_fname = '../data/hk_data/gt_lab_predictions_08052024.pkl'
# dump_fname = '../data/hk_data/gt_lab_predictions_08082024.pkl'
# lab_predictions = pickle.load(open(saved_fname, 'rb'))
# for _, id in enumerate(tqdm.tqdm(prediction_prompts.keys())):
#     lab_pred_prompt = prediction_prompts[id]['prediction_prompt']
#     if (id not in low_k_cohort) and (id not in female_cohort) and (id not in nonrenal_cohort):
#         continue
#     if id not in lab_predictions.keys():
#         lab_predictions[id] = {}
#     for model in MODELS:
#         if model not in lab_predictions[id].keys():
#             curr_tries = 0 # Predict the lab value directly from the EHR
#             success = False
#             while not success and curr_tries < num_tries:
#                 try:
#                     curr_tries += 1
#                     raw_prediction = prompt_llm(prediction_prompts[id]['prediction_prompt'], model=model)
#                     # Read in the prediction
#                     lab_after = prediction_prompts[id]['gt_lab_after']
#                     pred = read_lab_prediction(raw_prediction, model)
#                     print("ID: " + str(id) + " Model: " + str(model) + " GT Lab value: " + str(lab_after) + " Predicted Lab value: " + str(pred))
#                     success = True
#                     lab_predictions[id][model] = pred
#                 except Exception as e:
#                     wait = np.random.uniform(10, 50)
#                     print(f"Error ({e}), model={model}, try={curr_tries}, Waiting {wait:.1f}s before retrying...")
#                     time.sleep(wait)
#         pickle.dump(lab_predictions, open(dump_fname, 'wb'))
