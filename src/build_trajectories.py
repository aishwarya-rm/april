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
# Use the path of the notebook instead
notebook_path = Path().resolve()  # Gets the current working directory of the notebook
sys.path.append(str(notebook_path.parent))  # Go one level up, adjust if needed
from src.utils import find_sandwich

# Load the data

input_events = pd.read_csv("../data/inputevents.csv")
admissions = pd.read_csv("../data/admissions.csv")
patients = pd.read_csv("../data/patients.csv")
diagnoses = pd.read_csv("../data/diagnoses_icd.csv")
d_diagnoses = pd.read_csv("../data/d_icd_diagnoses.csv")
d_items = pd.read_csv("../data/d_items.csv")
subject_ids = pickle.load(open('../data/hk_data/hk_subject_ids.pkl', 'rb'))

filtered_chartevents = pd.read_csv("../data/filtered_chartevents.csv")

# Build trajectories for all patients, and include information that can tie a given trajectory back to the patient ID
def build_trajectories(subject_ids, fname, existing_trajectories):
    standard_features = ['age', 'gender', 'weight']
    lab_features = ['Heart Rate', 'Respiratory Rate', 'O2 saturation pulseoxymetry',
                    'Non Invasive Blood Pressure systolic',
                    'Non Invasive Blood Pressure diastolic', 'Creatinine (serum)', 'Creatinine (whole blood)',
                    'Anion gap',
                    'Ionized Calcium']
    med_features = ['NaCl 0.9%', 'Dextrose 5%', 'Propofol', 'Norepinephrine', 'Insulin - Regular',
                    'Fentanyl']  # Removed the last two, because most people had 0
    potassium_meds_ids = [225166]  # Potassium Chloride
    potassium_lab_ids = [227442]  # Serum potassium
    td = 4  # hours

    if len(existing_trajectories.keys()) == 0:
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_sum_rewards = []
        trajectory_ids = []
    else:
        trajectory_states = existing_trajectories['states'].copy()
        trajectory_actions = existing_trajectories['actions'].copy()
        trajectory_rewards = existing_trajectories['rewards'].copy()
        trajectory_sum_rewards = existing_trajectories['reward_sum'].copy()
        trajectory_ids = existing_trajectories['subject_ids'].copy()

    for _, id in enumerate(tqdm.tqdm(subject_ids)):  # End with med start time.
        if id in existing_trajectories['subject_ids']:
            continue
        else:
            kcl_meds_id = input_events[
                (input_events['subject_id'] == id) & input_events['itemid'].isin(potassium_meds_ids)]
            labs_id = filtered_chartevents[filtered_chartevents['subject_id'] == id]  # All labs observed
            hadm_id = labs_id['hadm_id'].iloc[0]  # This is with a very small subset of patients then?

            # Find the sandwich
            serum_k_labs_id = labs_id[labs_id['itemid'].isin(potassium_lab_ids)]
            med_dosage, med_starttime, med_endtime, lab_before, lab_time_before, lab_after, lab_time_after = find_sandwich(
                kcl_meds_id, serum_k_labs_id)

            if med_dosage is None:
                continue  # We couldn't find a sandwich for this patient.

            fmt = '%Y-%m-%d %H:%M:%S'
            med_st = datetime.strptime(med_starttime, fmt)
            med_et = datetime.strptime(med_endtime, fmt)
            lab_bt = datetime.strptime(lab_time_before, fmt)
            lab_at = datetime.strptime(lab_time_after, fmt)

            admittime = admissions[admissions['hadm_id'] == hadm_id]['admittime'].item()
            dischtime = admissions[admissions['hadm_id'] == hadm_id]['dischtime'].item()
            admittime_parsed = datetime.strptime(admittime, "%Y-%m-%d %H:%M:%S")
            dischtime_parsed = datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S")

            # Total number of hours of history that we will have in the trajectory.
            hours_history = ((
                                         med_st - admittime_parsed).total_seconds() / 3600)  # Total hours is the number of hours until the medicine was finished.
            if hours_history > 50 or hours_history < 0:  # Too long, too short
                continue
            print("Hours history: " + str(hours_history))

            if patients[patients['subject_id'] == id]['gender'].item() == 'F':
                gender = 'female'
            else:
                gender = 'male'
            age = patients[patients['subject_id'] == id]['anchor_age'].item()
            meds_by_id = input_events[(input_events['subject_id'] == id)]
            weight = meds_by_id.iloc[0]['patientweight']

            # Generate history
            patient_states = []
            patient_actions = []
            patient_rewards = []

            # Store possible information that can be moved across intervals (heart rate, respiratory rate, o2 saturation, sys bp, dias bp)
            lab_values_by_time = {}
            for i in range(0, int(hours_history // td) + 1):
                lab_values_by_time[i] = {}
                hours_in = int(i * td)
                start_interval = (admittime_parsed + timedelta(hours=hours_in))
                end_interval = (start_interval + timedelta(hours=td)).strftime("%Y-%m-%d %H:%M:%S")
                start_interval = start_interval.strftime("%Y-%m-%d %H:%M:%S")

                # Find all of the chartevents and input events in this window
                chartevents_window = filtered_chartevents[(filtered_chartevents['subject_id'] == id) & (
                        (filtered_chartevents['charttime'] >= start_interval) & (
                            filtered_chartevents['charttime'] <= end_interval))]

                labs_by_feature = {}
                for feat in lab_features:
                    labs_by_feature[feat] = []

                for index, row in chartevents_window.iterrows():
                    itemid = row['itemid']
                    lab_name = d_items[d_items['itemid'] == itemid]['label'].item()
                    if lab_name in labs_by_feature.keys():
                        labs_by_feature[lab_name].append(row['valuenum'])

                for lab in lab_features:
                    vals = labs_by_feature[lab]
                    if len(vals) == 0:
                        continue
                    else:
                        lab_values_by_time[i][lab] = np.mean(
                            vals)  # Aggregate all of the labs so that you can impute those.

            for i in range(0,
                           int(hours_history // td) + 1):  # From admittime to end of potassium administration (first four hours)
                hours_in = int(i * td)
                start_interval = (admittime_parsed + timedelta(hours=hours_in))
                end_interval = (start_interval + timedelta(hours=td)).strftime("%Y-%m-%d %H:%M:%S")
                start_interval = start_interval.strftime("%Y-%m-%d %H:%M:%S")

                state_representation = [age, int(gender == 'female'), weight]
                action_chosen = []

                chartevents_window = filtered_chartevents[(filtered_chartevents['subject_id'] == id) & (
                        (filtered_chartevents['charttime'] >= start_interval) & (
                            filtered_chartevents['charttime'] <= end_interval))]
                inputevents_window = input_events[(input_events['subject_id'] == id) & (
                        (input_events['starttime'] >= start_interval) & (
                            input_events['starttime'] <= end_interval))]

                labs_by_feature = {}
                for feat in lab_features:
                    labs_by_feature[feat] = []
                meds_by_feature = {}
                for feat in med_features:
                    meds_by_feature[feat] = []
                for index, row in chartevents_window.iterrows():
                    itemid = row['itemid']
                    lab_name = d_items[d_items['itemid'] == itemid]['label'].item()
                    if lab_name in labs_by_feature.keys():
                        labs_by_feature[lab_name].append(row['valuenum'])  # This is

                meds_patient = []
                for index, row in inputevents_window.iterrows():
                    itemid = row['itemid']
                    med_name = d_items[d_items['itemid'] == itemid]['label'].item()
                    if med_name in meds_by_feature.keys():
                        meds_by_feature[med_name].append(row['amount'])
                    if med_name == 'Potassium Chloride':  # This is the dosage of potassium chloride.
                        action_chosen.append(int(row['amount']))

                # Construct state representation/action_representation/reward
                for lab in lab_features:
                    vals = labs_by_feature[lab]
                    if len(vals) == 0:
                        time_keys = [i - 3, i + 3, i - 2, i + 2, i - 1, i + 1]
                        value = None
                        for k in time_keys:
                            if k in lab_values_by_time and lab in lab_values_by_time[k]:
                                value = lab_values_by_time[k][lab]  # Average all measurements for this individual.
                        if value is None:  # Still didn't find anything
                            state_representation.append(0)
                        else:
                            state_representation.append(value)
                    else:
                        state_representation.append(np.mean(vals))  # Mean of measured values.
                for med in med_features:
                    vals = meds_by_feature[med]
                    if len(vals) == 0:
                        state_representation.append(0)
                    else:
                        state_representation.append(np.mean(vals))

                if i == int(hours_history // td):  # This is the last timestep we have.
                    reward = int((lab_after >= 3.5) and (lab_after <= 5))
                else:
                    reward = 0

                patient_states.append(state_representation)
                if len(action_chosen) == 0:  # There is no positive dosage of potassium in this interval.
                    patient_actions.append(0)
                else:
                    patient_actions.append(
                        np.mean(action_chosen))  # There were >= 1 dosages of potassium in this interval.

                patient_rewards.append(reward)

            trajectory_states.append(patient_states)
            trajectory_actions.append(patient_actions)
            trajectory_rewards.append(patient_rewards)
            trajectory_sum_rewards.append(np.sum(patient_rewards))
            trajectory_ids.append(id)
            expert_trajectories = {'states': trajectory_states, 'actions': trajectory_actions,
                                   'rewards': trajectory_rewards,
                                   'reward_sum': trajectory_sum_rewards, 'subject_ids': trajectory_ids}
            pickle.dump(expert_trajectories, open(fname, 'wb'))  # Save the trajectories, and also return them.

    expert_trajectories = {'states': trajectory_states, 'actions': trajectory_actions, 'rewards': trajectory_rewards,
                           'reward_sum': trajectory_sum_rewards, 'subject_ids': trajectory_ids}
    pickle.dump(expert_trajectories, open(fname, 'wb'))  # Save the trajectories, and also return them.
    return expert_trajectories

# Male vs Female.
male_subject_ids = [] # 15460, target
female_subject_ids = [] # 11941, behavior
for _, id in enumerate(tqdm.tqdm(subject_ids)):
    # gender
    gender = patients[patients['subject_id']==id]['gender'].item()
    if gender == 'F':
        female_subject_ids.append(id)
    elif gender == 'M':
        male_subject_ids.append(id)


female_trajectories_v2 = build_trajectories(female_subject_ids[:2000], "trajectories/hk_female_06292025.pkl", female_trajectories) # Only doing this on 2k female trajectories, this just takes forever.
# male_trajectories_v2 = build_trajectories(male_subject_ids[:2000], "trajectories/hk_male_06292025.pkl", male_trajectories)



