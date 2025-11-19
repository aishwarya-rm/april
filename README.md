# TutorTest
This repository contains the code accompanying the paper:
“APRIL: Annotations for Policy evaluation with Reliable Inference from LLMs”

Venue: Machine Learning for Healthcare Symposium 2024

Authors: Aishwarya Mandyam, Kalyani Limaye, Barbara E. Engelhardt*, Emily Alsentzer *

# Overview
APRIL is a two stage framework used to generate counterfactual annotations using LLMs for medical tasks.

# Requirements
Python 3.9

# Datasets
Download the MIMIC-IV dataset (https://physionet.org/content/mimiciv/3.1/) after going through the appropriate certifications. Place all non-ICU csv files in data/. 

# Preprocessing
To create the contextual bandits, run 
```python
python src/build_datasets.py
```

To learn the corresponding behavior and target policies, run
```python
python src/infer_policies.py
```

# Query Counterfactual Annotations
To query counterfactual annotations, first update LLM access details including access keys and LLM URLs in src/utils.py. Then, run 
```python
python experiments/solicit_ounterfactual_annotations.py
```

# Run Off-policy evaluation
After generating the datasets, run off-policy evaluation using:
```python
python experiments/run_ope.py
```

# Contact
For questions or issues:

Contact: Aishwarya Mandyam (am2@stanford.edu)

