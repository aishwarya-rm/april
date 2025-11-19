import sys
from pathlib import Path
sys.path.append((str(Path(__file__).absolute().parent.parent)))
from src.llm_prompting import prompt_llm, read_lab_prediction, read_lab_summary
import pickle
import tqdm
import time
import numpy as np
from src.utils import MODELS

def solicit_counterfactual_annotations(annotation_prompts, subject_ids, dump_fname, task):
    num_tries = 3
    counterfactual_annotations = {}
    for _, id in enumerate(tqdm.tqdm(annotation_prompts.keys())):
        if id not in subject_ids:
            continue
        if task == 'potassium':
            possible_ca_dosages = [0, 10, 20, 40]
        elif task == 'sodium':
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
                            success = True  
                        except Exception as e:
                            wait = np.random.uniform(10, 50)
                            print(f"Error ({e}), model={model}, try={curr_tries}, Waiting {wait:.1f}s before retrying...")
                            time.sleep(wait)
                pickle.dump(counterfactual_annotations, open(dump_fname, 'wb'))

if __name__ == 'main':
    potassium_subject_ids = pickle.load(open('../data/potassium_ids.pkl', 'rb'))
    potassium_prompts = pickle.load(open('../prompts/potassium_prediction_prompts.pkl', 'rb'))
    solicit_counterfactual_annotations(potassium_prompts, potassium_subject_ids, "../data/potassium_annotations.pkl", "potassium")

    sodium_subject_ids = pickle.load(open('../data/sodium_ids.pkl', 'rb'))
    sodium_prompts = pickle.load(open("../prompts/sodium_prediction_prompts.pkl", 'rb'))
    solicit_counterfactual_annotations(sodium_prompts, sodium_subject_ids, "../data/sodium_annotations.pkl", "sodium")






