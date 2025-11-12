import sys
from pathlib import Path
sys.path.append((str(Path(__file__).absolute().parent.parent)))
from src.llm_prompting import prompt_llm, read_lab_prediction, read_lab_summary
import pickle
import tqdm
import time
import numpy as np
from src.utils import MODELS

# annotation_prompts = pickle.load(open('../data/hk_data/ca_prompts_06082025.pkl', 'rb')) # EHR --> Lab Value
annotation_prompts = pickle.load(open('../data/hn_data/ca_prompts_08012025.pkl', 'rb')) # EHR --> Lab Value

# Subject IDS we care about
# low_k_ids = pickle.load(open('../notebooks/trajectories/hk_low_dosage.pkl', 'rb'))
# nonrenal_ids = pickle.load(open('../notebooks/trajectories/hk_nonrenal.pkl', 'rb'))

# Subject IDS we care about
low_na_ids = pickle.load(open('../data/hn_data/low_na_ids.pkl', 'rb'))
female_ids = pickle.load(open('../data/hn_data/female_ids.pkl', 'rb'))
no_cirrhosis_ids = pickle.load(open('../data/hn_data/no_cirrhosis_ids.pkl', 'rb'))

saved_fname = '../data/hn_data/all_annotations_08082025.pkl'
dump_fname = '../data/hn_data/all_annotations_08102025.pkl'

# Counterfactual annotations should be for female subject ids
num_tries = 3
counterfactual_annotations = pickle.load(open(saved_fname, 'rb'))
for _, id in enumerate(tqdm.tqdm(annotation_prompts.keys())):
    if id in counterfactual_annotations.keys(): # Just keep generating all possible counterfactual annotations.
        continue
    if (id not in low_na_ids) and (id not in no_cirrhosis_ids) and (id not in female_ids): # Keep generating for behavior trajectories that we expect to see.
        continue
    # possible_ca_dosages = [0, 10, 20, 40]
    possible_ca_dosages = [0, 100, 200, 300, 400, 500]
    counterfactual_annotations[id] = {}
    for model in MODELS:
        counterfactual_annotations[id][model] = {}
        for dosage in possible_ca_dosages:
            if dosage in annotation_prompts[id].keys(): # Do we have a prompt for this.
                annotation_prompt = annotation_prompts[id][dosage]
                curr_tries = 0
                success = False
                while not success and curr_tries < num_tries:
                    try:
                        curr_tries += 1
                        raw_prediction = prompt_llm(annotation_prompt, model=model)
                        pred = read_lab_prediction(raw_prediction, model)
                        print("ID: " + str(id) + " Model: " + str(model) + " Dosage: " + str(dosage) + " Predicted Lab value: " + str(pred))
                        counterfactual_annotations[id][model][dosage] = pred
                        success = True  # We really do need all annotations (probably around 200).
                    except Exception as e:
                        wait = np.random.uniform(10, 50)
                        print(f"Error ({e}), model={model}, try={curr_tries}, Waiting {wait:.1f}s before retrying...")
                        time.sleep(wait)
            pickle.dump(counterfactual_annotations, open(dump_fname, 'wb')) # Getting more annotations.




