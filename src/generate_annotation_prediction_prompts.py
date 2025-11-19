import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import json
from datetime import datetime, timedelta
from src.utils import find_sandwich


input_events = pd.read_csv("../data/inputevents.csv")
admissions = pd.read_csv("../data/admissions.csv")
patients = pd.read_csv("../data/patients.csv")
diagnoses = pd.read_csv("../data/diagnoses_icd.csv")
d_diagnoses = pd.read_csv("../data/d_icd_diagnoses.csv")
d_items = pd.read_csv("../data/d_items.csv")
chartevents = pd.read_csv('../data/chartevents.csv')

def generate_prediction_prompts(subject_ids, dump_fname, task):
    prediction_prompts = {} # 'prompt to predict lab value from EHR data', 'gt_lab'. 'auxiliary information to help with lab prediction'
    deltas_before = []
    deltas_after = []

    if task == 'potassium':
        target_meds_ids = [225166] # Potassium Chloride
        target_lab_ids = [227442] # Serum potassium
    elif task == 'sodium':
        target_meds_ids = [ 225161] #221211, 225926,,229861,
        target_lab_ids = [220645] # Serum sodium

    for _, id in enumerate(tqdm.tqdm(subject_ids)):
        kcl_meds_id = input_events[(input_events['subject_id']==id) & input_events['itemid'].isin(target_meds_ids)] # All potassium administrations
        labs_id = chartevents[chartevents['subject_id'] == id] # All labs observed
        hadm_id = labs_id['hadm_id'].iloc[0]

        # Find the sandwich
        serum_k_labs_id = labs_id[labs_id['itemid'].isin(target_lab_ids)]
        med_dosage, med_starttime, med_endtime, lab_before, lab_time_before, lab_after, lab_time_after = find_sandwich(kcl_meds_id, serum_k_labs_id)
        if med_dosage is None: 
            continue # We couldn't find a sandwich for this patient. 
            
        fmt = '%Y-%m-%d %H:%M:%S'
        med_st = datetime.strptime(med_starttime, fmt)
        med_et = datetime.strptime(med_endtime, fmt)
        lab_bt = datetime.strptime(lab_time_before, fmt)
        lab_at = datetime.strptime(lab_time_after, fmt)
        
        # Get the difference in hours
        delta_before = (med_st - lab_bt).total_seconds() / 3600 # Time between med administration and the previous K lab
        delta_after = (lab_at - med_et).total_seconds() / 3600  # Time when lab is taken after finishing administration. 
        deltas_before.append(delta_before)
        deltas_after.append(delta_after)
        if delta_before > 12 or delta_after > 12: # there was no lab taken in the previous 12 hours or in the 12 hours post
            continue

        # Construct the patient summary
        admittime = admissions[admissions['hadm_id'] == hadm_id]['admittime'].item()
        dischtime = admissions[admissions['hadm_id'] == hadm_id]['dischtime'].item()
        admittime_parsed = datetime.strptime(admittime, "%Y-%m-%d %H:%M:%S")
        dischtime_parsed = datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S")
        
        hours_history = ((med_et - admittime_parsed).total_seconds() / 3600) // 4 # Total hours is the number of hours until the medicine was finished. 
        if hours_history <= 0 and hours_history > 100:
            continue
        total_hours = (dischtime_parsed - admittime_parsed).total_seconds() / 3600
        general_information = "You are interested in the task of predicting a patient's serum potassium level as measured from a blood sample after administering a dose of potassium chloride through an intravenous line (IV). What follows is a description of the events that occured in the last four hours of the patient's hospital stay that will help you predict the serum potassium level. These details include [[ ## Reason for Admission ## ]], [[ ## Static Covariates ## ]], [[ ## Labs/Vitals/Procedures ## ]], and [[ ## Medications ## ]]. \n\n"
        uptodate_information = "UpToDate, a relevant health resource, suggests that the most important features to rely on to predict the outcome of potassium repletion include gastrointestinal/renal losses, renal disease, other electrolyte abnormalities, serum potassium labs, cardiovascular risk factors, kidney function, and the presence of concurrent medical conditions such as diabetic ketoacidosis, cirrhosis, and heart failure. \n\n"
        
        gender_reported = patients[patients['subject_id']==id]['gender'].item()
        if gender_reported == 'F':
            gender = 'female'
        else:
            gender = 'male'
        age = patients[patients['subject_id']==id]['anchor_age'].item()
        
        meds_by_id = input_events[(input_events['subject_id']==id)]
        diagnoses_patient = diagnoses[diagnoses['hadm_id'] == hadm_id]
        
        diagnoses_list = []
        for code, version in zip(diagnoses_patient['icd_code'], diagnoses_patient['icd_version']):
            row = d_diagnoses[(d_diagnoses['icd_code'] == code) & (d_diagnoses['icd_version'] == version)]
            diagnoses_list.append(row['long_title'].item())
        
        # Patient weight
        weight = meds_by_id.iloc[0]['patientweight']
        
        # Generate history
        timestep = hours_history - 1 # What if we made it 6 hours. 
        
        admittime_reported = admissions[admissions['hadm_id'] == hadm_id]['admittime'].item()
        reason_admission = admissions[admissions['hadm_id'] == hadm_id]['admission_type'].item()
        
        static_covariates = f"[[ ## Reason for Admission ## ]] \n\n The patient was admitted at {admittime_reported} and they were admitted for \n{reason_admission.lower()}. \n\n"
        static_covariates += f"[[ ### Static Covariates ## ]]: \n\n The patient is a {gender} who is {age} years old, weighs {weight} kgs, and has the following comorbidites: \n {chr(10).join(diagnoses_list)}.\n\n " 

        hours_in = timestep * 4 # Each interval is 4 hours
        start_time = (admittime_parsed + timedelta(hours=hours_in))
        end_time = (start_time + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
        start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")

        chartevents_window = chartevents[(chartevents['subject_id'] == id) & ((chartevents['charttime'] >= start_time) & (chartevents['charttime'] <= end_time))]
        inputevents_window = input_events[(input_events['subject_id']==id) & ((input_events['starttime'] >= start_time) & (input_events['starttime'] <= end_time))]
        labs_and_meds = f""
        labs = []
        for index, row in chartevents_window.iterrows():
            itemid = row['itemid']
            lab_name = d_items[d_items['itemid'] == itemid]['label'].item()
            lab_time = row['charttime']
            if str(row['valueuom']) == 'nan':
                labs.append(f"{lab_name} measured {row['value']}, at {lab_time}")
            else:
                labs.append(f"{lab_name} measured {row['value']} {row['valueuom']} at {lab_time}")
        if len(labs) == 0:
            labs_and_meds += f"[[ ### Labs/Vitals/Procedures ## ]]: \n\n The patient did not have any labs or procedures in the last four hours. \n\n"
        else:
            labs_and_meds += f"[[ ### Labs/Vitals/Procedures ## ]]: \n Here is a list of measured lab values and procedures for the patient during the last four hours: \n {chr(10).join(labs)}. \n\n"
        
        meds = []
        for index, row in inputevents_window.iterrows():
            itemid = row['itemid']
            med_name = d_items[d_items['itemid'] == itemid]['label'].item()
            med_starttime = row['starttime']
            med_endtime = row['endtime']
            meds.append(f"{row['amount']} {row['amountuom']} of {med_name}, started at {med_starttime} and ended at {med_endtime}")
        if len(meds) == 0:
            labs_and_meds += f"[[ ### Medications ## ]]: \n The patient received no medication in the last four hours. \n\n"
        else:
            labs_and_meds += f"[[ ### Medications ## ]]: \n Here is a list of medications the patient received during the last four hours: \n {chr(10).join(meds)}. \n\n"

        task_goal = f"Recall that your goal is to predict a patient's serum potassium lab after administering a dose of potassium through an IV. Remember that the patient's latest serum potassium lab value is {lab_before} mEq/L.\n\n"
        example_file = '{"predicted_lab_value": "[]", "justification": "[]."}'
        task_information = f" The patient will receive a total dosage of {int(med_dosage)} mEq of potassium through an IV drip. The drip will start at {med_starttime} and end at {med_endtime}. What will the patient's blood potassium level be around {int(delta_after)} hour(s) after receiving this dose of IV potassium? Examine the patient’s clinical record description, which includes labs, vitals, comorbidities, medications administered (especially potassium doses), and the timing of those details. Then, based on all available relevant factors, determine the most likely numeric serum potassium level following potassium administration. Phrase your response as a JSON file. In particular, the file should have two keys. One for the predicted lab value, in mEq/L, titled predicted_lab_value, and one for the justification titled justification. An example of this type of file is the following: {example_file}. In this example, insert your predicted lab value in the list for the first key, and the justification in the list for the second key. Remember not to include units in the prediction and make sure that the prediction is a single number and not a range.\n\n"
        prediction_prompts[id] = {}
        prediction_prompts[id]['prediction_prompt'] = general_information + static_covariates + labs_and_meds + uptodate_information + task_goal + task_information
        prediction_prompts[id]['gt_lab_after'] = lab_after
        prediction_prompts[id]['gt_med'] = int(med_dosage)
        prediction_prompts[id]['med_start'] = med_starttime
        prediction_prompts[id]['med_end'] = med_endtime
        prediction_prompts[id]['delta_after'] = delta_after
        prediction_prompts[id]['gt_lab_before'] = lab_before
        pickle.dump(prediction_prompts, open(dump_fname, 'wb'))
    

if __name__ == 'main':
    potassium_subject_ids = pickle.load(open('../data/potassium_ids.pkl', 'rb'))
    sodium_subject_ids = pickle.load(open('../data/sodium_ids.pkl', 'rb'))

    generate_prediction_prompts(potassium_subject_ids, "../prompts/potassium_prediction_prompts.pkl", "potassium")
    generate_prediction_prompts(sodium_subject_ids, "../prompts/sodium_prediction_prompts.pkl", "sodium")