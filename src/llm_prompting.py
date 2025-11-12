'''
All of the functions that you need to run a particular experiment (so prompting for summaries, prompting for counterfactual annotations)
'''
from src.utils import MODEL_TO_URL, HEADERS
import pickle
import json
import requests
import ast

def prompt_llm(prompt, model, **kwargs):
    '''
    Summarizes the patient state by querying the LLM, and sets temperature to 0 whenever possible.
    :param patient_desc: 
    :param model: String, should be model name (as listed in utils.py)
    :return: 
    '''

    if model in ['o1', 'o3-mini']:
        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
    elif model in ['gpt-4', 'gpt-4o-mini']:
        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
        })
    elif model == 'gemini':
        payload = json.dumps({
            "contents": {
                "role": "user",
                "parts": {
                    "text": prompt
                }
            },
            "safety_settings": {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            "generation_config": {
                "temperature": 0.0,
                "topP": 0.8,
                "topK": 40
            }
        })
    elif model == 'claude-3.7':
        payload = json.dumps({
            "model_id": "arn:aws:bedrock:us-west-2:679683451337:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "prompt_text": prompt,
            "temperature": 0
        })
    response = requests.request("POST", MODEL_TO_URL[model], headers=HEADERS, data=payload, timeout=40)
    return response.text

def read_lab_prediction(prediction_json, model):
    if model == 'gpt-4o-mini':
        try:
            pred = float(json.loads(json.loads(prediction_json)['choices'][0]['message']['content'].split('json')[1].split('```')[0])['predicted_lab_value'])
        except:
            pred = float(json.loads(json.loads(prediction_json)['choices'][0]['message']['content'].split('json')[1].split('```')[0])['predicted_lab_value'][0])
    elif model in ['o1', 'gpt-4']:
        try:
            pred = float(json.loads(json.loads(prediction_json)['choices'][0]['message']['content'])['predicted_lab_value'])
        except:
            pred = float(json.loads(json.loads(prediction_json)['choices'][0]['message']['content'])['predicted_lab_value'][0])
    elif model == 'o3-mini':
        prediction_string = json.loads(json.loads(prediction_json)['choices'][0]['message']['content'])['predicted_lab_value']
        try:
            pred = float(prediction_string) # It isn't in a list
        except:
            try:
                pred = float(ast.literal_eval(prediction_string)[0]) # It is in a list that is a string
            except:
                pred = float(prediction_string[0]) # It is just a list already

    elif model in ['gemini']:
        json_text = []
        desc = json.loads(prediction_json)
        for p in range(len(desc)):
            json_text.append(desc[p]['candidates'][0]['content']['parts'][0]['text'])
        pred = float(json.loads(''.join(json_text).split('json')[1].split('```')[0])['predicted_lab_value'][0])
    elif model == 'claude-3.7':
        pred_class = json.loads(json.loads(prediction_json)['content'][0]['text'].split('json')[1].split('```')[0])[
                         'predicted_lab_value']
        if isinstance(pred_class, list):
            pred = float(pred_class[0])
        else:
            pred = float(pred_class)
    return pred

def read_lab_summary(summary_json, model):
    if model in ['o1', 'gpt-4', 'gpt-4o-mini', 'o3-mini']:
        desc = json.loads(summary_json)['choices'][0]['message']['content']
    elif model == 'gemini':
        desc_json = json.loads(summary_json)
        summary = []
        for p in range(len(desc_json)):
            summary.append(desc_json[p]['candidates'][0]['content']['parts'][0]['text'])
        desc = ''.join(summary)
    elif model == 'claude-3.7':
        desc = json.loads(summary_json)['content'][0]['text']
    return desc





