import numpy as np
import pickle
import tqdm
import pandas as pd
import json
from datetime import datetime, timedelta
import ipdb
import pickle
import sys
from pathlib import Path
notebook_path = Path().resolve()  
sys.path.append(str(notebook_path.parent))
from src.utils import find_sandwich

# Load the data
input_events = pd.read_csv("../data/inputevents.csv")
admissions = pd.read_csv("../data/admissions.csv")
patients = pd.read_csv("../data/patients.csv")
diagnoses = pd.read_csv("../data/diagnoses_icd.csv")
d_diagnoses = pd.read_csv("../data/d_icd_diagnoses.csv")
d_items = pd.read_csv("../data/d_items.csv")
chartevents = pd.read_csv('../data/chartevents.csv')
subject_ids = pickle.load(open('../data/hk_data/hk_subject_ids.pkl', 'rb'))

# Build contextual bandit dataset 
def build_cb_dataset(subject_ids, fname, task):
    standard_features = ['age', 'gender', 'weight']
    lab_features = ['Heart Rate', 'Respiratory Rate', 'O2 saturation pulseoxymetry', 'Non Invasive Blood Pressure systolic',
                    'Non Invasive Blood Pressure diastolic', 'Creatinine (serum)']
    med_features = ['NaCl 0.9%', 'Dextrose 5%', 'Propofol', 'Norepinephrine', 'Insulin - Regular', 'Fentanyl'] # Removed the last two, because most people had 0
    
    if task == 'potassium':
        target_meds_ids = [225166] # Potassium Chloride
        target_lab_ids = [227442] # Serum potassium
    elif task == 'sodium':
        target_meds_ids = [ 225161] #221211, 225926,,229861,
        target_lab_ids = [220645] # Serum sodium
    td = 4  # hours

    contexts = []
    actions = []
    clinical_features = []
    ids = []
    
    for _, id in enumerate(tqdm.tqdm(subject_ids)): # End with med start time.
        kcl_meds_id = input_events[(input_events['subject_id'] == id) & input_events['itemid'].isin(target_meds_ids)]
        labs_id = chartevents[chartevents['subject_id'] == id]  # All labs observed
        hadm_id = labs_id['hadm_id'].iloc[0] 
    
        # Find the sandwich 
        serum_target_labs_id = labs_id[labs_id['itemid'].isin(target_lab_ids)]
        med_dosage, med_starttime, med_endtime, lab_before, lab_time_before, lab_after, lab_time_after = find_sandwich(
            kcl_meds_id, serum_target_labs_id)
        
        if med_dosage is None:
            continue  # We couldn't find a sandwich for this patient.
    
        fmt = '%Y-%m-%d %H:%M:%S'
        med_st = datetime.strptime(med_starttime, fmt)
        med_et = datetime.strptime(med_endtime, fmt)
        lab_bt = datetime.strptime(lab_time_before, fmt)
        lab_at = datetime.strptime(lab_time_after, fmt)
        
        if patients[patients['subject_id'] == id]['gender'].item() == 'F': # identify the gender
            gender = 'female'
        else:
            gender = 'male'
        age = patients[patients['subject_id'] == id]['anchor_age'].item()
        meds_by_id = input_events[(input_events['subject_id'] == id)]
        weight = meds_by_id.iloc[0]['patientweight']
    
        start_interval = (med_st - timedelta(hours=4)) # start four hours before the drug was given 
        start_interval = start_interval.strftime("%Y-%m-%d %H:%M:%S")
        end_interval = med_st # end at the time the drug was given.
        end_interval = end_interval.strftime("%Y-%m-%d %H:%M:%S")
    
        # construct the patient context, which contains information for all features in the four hours before the drug was given. 
        state_representation = [age, int(gender == 'female'), weight]
        
        chartevents_window = chartevents[(chartevents['subject_id'] == id) & (
                    (chartevents['charttime'] >= start_interval) & (chartevents['charttime'] <= end_interval))]
        inputevents_window = input_events[(input_events['subject_id'] == id) & (
                    (input_events['starttime'] >= start_interval) & (input_events['starttime'] <= end_interval))]
    
        labs_by_feature = {feat: [] for feat in lab_features}
        meds_by_feature = {feat: [] for feat in med_features}
        
        for index, row in chartevents_window.iterrows(): # gather all labs
            itemid = row['itemid']
            lab_name = d_items[d_items['itemid'] == itemid]['label'].item()
            if lab_name in labs_by_feature.keys():
                labs_by_feature[lab_name].append(row['valuenum'])
    
        for index, row in inputevents_window.iterrows(): # gather all medicines
            itemid = row['itemid']
            med_name = d_items[d_items['itemid'] == itemid]['label'].item()
            if med_name in meds_by_feature.keys():
                meds_by_feature[med_name].append(row['amount'])
    
        # Construct state representation/action_representation/reward
        for lab in lab_features:
            vals = labs_by_feature[lab]
            state_representation.append(np.nanmean(vals)) # Mean of measured values. 
        for med in med_features:
            vals = meds_by_feature[med]
            if len(vals) == 0:
                state_representation.append(0)
            else:
                state_representation.append(np.mean(vals))
    
        contexts.append(state_representation)
        actions.append(med_dosage)
        clinical_features.append(lab_after)
        ids.append(id)
        behavior_dataset = {'contexts': contexts, 'actions': actions, 'clinical_features': clinical_features, 'subject_ids':ids}
        pickle.dump(behavior_dataset, open(fname, 'wb'))
        
    pickle.dump(behavior_dataset, open(fname, 'wb'))
    return behavior_dataset

def define_patient_cohorts(ids, task):
    male_subject_ids = [] 
    female_subject_ids = [] 
    comorbidity_ids = []
    no_comorbidity_ids = []
    low_dosage_ids = []
    high_dosage_ids = []

    for _, id in enumerate(tqdm.tqdm(subject_ids)):
        # gender
        gender = patients[patients['subject_id']==id]['gender'].item()
        if gender == 'F':
            female_subject_ids.append(id)
        elif gender == 'M':
            male_subject_ids.append(id)
        
        # comorbidity
        if task == 'sodium':
            cirr_items = []
            for item in d_diagnoses['long_title']:
                text_lower = item.lower()
                if (
                    'cirrhosis' in text_lower
                    or 'cirrhotic' in text_lower
                    or 'alcoholic cirrhosis' in text_lower
                    or 'biliary cirrhosis' in text_lower
                    or 'portal cirrhosis' in text_lower
                    or 'hepatic cirrhosis' in text_lower
                ):
                    cirr_items.append(item)
            hadm_id = chartevents[chartevents['subject_id'] == id]['hadm_id'].iloc[0]
            diagnoses_patient = diagnoses[diagnoses['hadm_id'] == hadm_id]

            diagnoses_titles = pd.merge(diagnoses_patient, d_diagnoses, on=['icd_code', 'icd_version'], how='inner')['long_title']
            intersection_cirrhosis = list(set(diagnoses_titles) & set(cirr_items))

            if len(intersection_cirrhosis) > 0:
                comorbidity_ids.append(id)
            else:
                no_comorbidity_ids.append(id)
        elif task == 'potssium':
            renal_disease_items = []
            for item in d_diagnoses['long_title']:
                if 'renal disease' in item.lower(): # TODO: iron administration (IV)
                    renal_disease_items.append(item)
                
            hadm_id = chartevents[chartevents['subject_id'] == id]['hadm_id'].iloc[0]
            diagnoses_patient = diagnoses[diagnoses['hadm_id'] == hadm_id]

            diagnoses_titles = pd.merge(diagnoses_patient, d_diagnoses, on=['icd_code', 'icd_version'], how='inner')['long_title']
            intersection_renal = list(set(diagnoses_titles) & set(renal_disease_items))

            if len(intersection_renal) > 0:
                comorbidity_ids.append(id)
            else:
                no_comorbidity_ids.append(id)

        # dosage
        if task == 'sodium':
            sodium_meds_ids = [225161]
            sodium_labs_ids = [220645]
            median_dosage = 174
            na_meds_id = input_events[(input_events['subject_id'] == id) & input_events['itemid'].isin(sodium_meds_ids)]
            labs_id = chartevents[chartevents['subject_id'] == id]  # All labs observed
            hadm_id = labs_id['hadm_id'].iloc[0] 

            # Find the sandwich
            serum_k_labs_id = labs_id[labs_id['itemid'].isin(sodium_labs_ids)]
            med_dosage, med_starttime, med_endtime, lab_before, lab_time_before, lab_after, lab_time_after = find_sandwich(kcl_meds_id, serum_k_labs_id)
            if med_dosage is not None and med_dosage <= median_dosage:
                low_dosage_ids.append(id)
            else:
                high_dosage_ids.append(id)


        elif task == 'potassium':
            potassium_meds_ids = [225166] # Potassium Chloride
            potassium_lab_ids =  [227442]
            median_dosage = 20.000000596046448
            kcl_meds_id = input_events[(input_events['subject_id'] == id) & input_events['itemid'].isin(potassium_meds_ids)]
            labs_id = chartevents[chartevents['subject_id'] == id]  # All labs observed
            hadm_id = labs_id['hadm_id'].iloc[0] 

            # Find the sandwich
            serum_k_labs_id = labs_id[labs_id['itemid'].isin(potassium_lab_ids)]
            med_dosage, med_starttime, med_endtime, lab_before, lab_time_before, lab_after, lab_time_after = find_sandwich(kcl_meds_id, serum_k_labs_id)
            if med_dosage is not None and med_dosage <= median_dosage:
                low_dosage_ids.append(id)
            else:
                high_dosage_ids.append(id)

    return (male_subject_ids, female_subject_ids), (comorbidity_ids, no_comorbidity_ids), (low_dosage_ids, high_dosage_ids)

        

# Define cohorts
potassium_subject_ids = pickle.load(open('../data/potassium_ids', 'rb'))
sodium_subject_ids = pickle.load(open('../data/sodium_ids.pkl', 'rb'))

(male_subject_ids, female_subject_ids), (comorbidity_ids, no_comorbidity_ids), (low_dosage_ids, high_dosage_ids) = define_patient_cohorts(potassium_subject_ids, task='potassium')

# Construct the contextual bandit datasets
cohorts = [male_subject_ids, female_subject_ids, comorbidity_ids, no_comorbidity_ids, low_dosage_ids, high_dosage_ids]
cohort_names = ["male_potassium.pkl", "female_potassium.pkl", "comorbidity_potassium.pkl", "no_comorbidity_potassium.pkl", "low_dosage_potassium.pkl", "high_dosage_potassium.pkl"]
dir = "../data/cohorts/"
for c_ids, name in zip(cohorts, cohort_names):
    dataset = build_cb_dataset(c_ids, dir + name)

(male_subject_ids, female_subject_ids), (comorbidity_ids, no_comorbidity_ids), (low_dosage_ids, high_dosage_ids) = define_patient_cohorts(sodium_subject_ids, task='sodium')
cohorts = [male_subject_ids, female_subject_ids, comorbidity_ids, no_comorbidity_ids, low_dosage_ids, high_dosage_ids]
cohort_names = ["male_sodium.pkl", "female_sodium.pkl", "comorbidity_sodium.pkl", "no_comorbidity_sodiumpkl", "low_dosage_sodium.pkl", "high_dosage_sodium.pkl"]
dir = "../data/cohorts/"
for c_ids, name in zip(cohorts, cohort_names):
    dataset = build_cb_dataset(c_ids, dir + name)



