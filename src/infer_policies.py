'''
Uses behavior cloning to infer the target (\pi_e) and behavior (\pi_b) policies from historical data.
'''
import sys
from pathlib import Path
sys.path.append((str(Path(__file__).absolute().parent.parent)))

import numpy as np
import tqdm
import torch
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pickle
from itertools import chain

np.random.seed(42)
torch.manual_seed(42)
NUM_ACTIONS=4
class DiscretePolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_sizes=[20, 20]):
        super(DiscretePolicyNetwork, self).__init__()
        layers = []
        input_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, num_actions)) # logits over actions
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)  # raw logits
        return torch.softmax(logits, dim=-1)  # Probability distribution

def encode_actions(raw_actions):
    num_classes = 4 # 0, 10, 20, 40
    encoded_actions = []
    for traj in raw_actions:
        for act in traj:
            act = round(act, -1)
            if act == 0:
                encoded_actions.append(0)
            elif act == 10:
                encoded_actions.append(1)
            elif act == 20:
                encoded_actions.append(2)
            elif act >= 30: # Some actions are 30, but most are 40 or over.
                encoded_actions.append(3)
            else:
                print(str(act))
    return np.asarray(encoded_actions)

def encode_cb_actions_hk(raw_actions):
    num_classes = 4 # 0, 10, 20, 40
    encoded_actions = []
    for act in raw_actions:
        act = round(act, -1)
        if act == 0:
            encoded_actions.append(0)
        elif act == 10:
            encoded_actions.append(1)
        elif act == 20:
            encoded_actions.append(2)
        elif act >= 30:  # Some actions are 30, but most are 40 or over.
            encoded_actions.append(3)
        else:
            print(str(act))
    return np.asarray(encoded_actions)

def encode_cb_actions_hn(raw_actions):
    num_classes = 6 # 0, 100, 200, 300, 400, 500
    encoded_actions = []
    for act in raw_actions:
        act = round(act, -2)
        if act == 0:
            encoded_actions.append(0)
        elif act == 100:
            encoded_actions.append(1)
        elif act == 200:
            encoded_actions.append(2)
        elif act == 300:
            encoded_actions.append(3)
        elif act == 400:
            encoded_actions.append(4)
        elif act >= 500:
            encoded_actions.append(5)
        else:
            print(str(act))
    return np.asarray(encoded_actions)


def train_discrete_policy(states, action_indices, task, state_dim, epochs, batch_size, lr, entropy_coeff):
    # Split into training and validation sets
    train_states, val_states, train_actions, val_actions = train_test_split(
        states, action_indices, test_size=0.2, random_state=42)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(train_states, dtype=torch.float32),
        torch.tensor(train_actions, dtype=torch.long)
    )
    val_states_tensor = torch.tensor(val_states, dtype=torch.float32)
    val_actions_tensor = torch.tensor(val_actions, dtype=torch.long)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if task == 'hn':
        policy = DiscretePolicyNetwork(state_dim, num_actions=6)
    else:
        policy = DiscretePolicyNetwork(state_dim, num_actions=4)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm.tqdm(range(epochs)):
        policy.train()
        epoch_losses = []
        epoch_accuracies = []

        for batch_states, batch_action_indices in loader:
            logits = policy(batch_states)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1).mean()
            loss = loss_fn(logits, batch_action_indices) - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            preds = logits.argmax(dim=-1)
            acc = (preds == batch_action_indices).float().mean().item()
            epoch_accuracies.append(acc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            # Evaluate on validation set every 10 epochs or last epoch
            policy.eval()
            with torch.no_grad():
                val_logits = policy(val_states_tensor)
                val_loss = loss_fn(val_logits, val_actions_tensor).item()
                val_preds = val_logits.argmax(dim=-1)
                val_accuracy = (val_preds == val_actions_tensor).float().mean().item()

            print(
                f"Epoch {epoch:03d} | Train Loss: {np.mean(epoch_losses):.4f}, Train Acc: {np.mean(epoch_accuracies):.3f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.3f}")

    return policy, val_loss, val_accuracy

def normalize_data(data):
    flattened_data = list(chain.from_iterable(data))  # List[List[float]]
    stacked = np.vstack(flattened_data)  # shape: (num_samples, num_features)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (stacked - mean)/std_safe, mean, std_safe

def normalize_cb_data(data):
    data = np.nan_to_num(data, nan=0.0) # Set all nans to 0.
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (data - mean) / std_safe, mean, std_safe

def learn_policy(trajectory_fname, policy_fname, metric_fname, epochs=30, batch_size=64, lr=1e-3, entropy_coeff=0.01, state_dim=18):
    print("LR: " + str(lr) + " Epochs: " + str(epochs) + " Entropy Coefficient: " + str(entropy_coeff))
    trajectories = pickle.load(open(trajectory_fname, 'rb'))
    states, mean, std = normalize_data(trajectories['states'])
    actions = encode_actions(trajectories['actions'])

    policy, val_loss, val_accuracy = train_discrete_policy(
        states, actions, state_dim=state_dim,
        epochs=epochs, batch_size=batch_size, lr=lr, entropy_coeff=entropy_coeff
    )

    torch.save(policy.state_dict(), policy_fname)
    np.save(metric_fname + "mean.npy", mean)
    np.save(metric_fname + "std.npy", std)

    print("Policy saved")
    return mean, std

def learn_cb_policy(dataset_fname, policy_fname, metric_fname, epochs=30, batch_size=20, lr=1e-3, entropy_coeff=0.01, state_dim=15, task='hk'):
    print("LR: " + str(lr) + " Epochs: " + str(epochs) + " Entropy Coefficient: " + str(entropy_coeff))
    behavior_dataset = pickle.load(open(dataset_fname, 'rb'))
    train_states, mean, std = normalize_cb_data(behavior_dataset['contexts'])
    if task == 'hk':
        train_actions = encode_cb_actions_hk(behavior_dataset['actions'])
    elif task == 'hn':
        train_actions = encode_cb_actions_hn(behavior_dataset['actions'])

    policy, val_loss, val_accuracy = train_discrete_policy(
        train_states, train_actions, task, state_dim=state_dim,
        epochs=epochs, batch_size=batch_size, lr=lr, entropy_coeff=entropy_coeff
    )

    torch.save(policy.state_dict(), policy_fname)
    np.save(metric_fname + "mean.npy", mean)
    np.save(metric_fname + "std.npy", std)

    print("Policy saved")
    return mean, std

if __name__ == "__main__":
    '''
    Actually learns the policies. 
    '''
    # Gender HK
    mean, std = learn_cb_policy("../notebooks/trajectories/male_cb_07172025.pkl", "../policies/hk_male_cb.pth", "../policies/hk_male_cb_", lr=7e-4, epochs=35)
    mean, std = learn_cb_policy("../notebooks/trajectories/female_cb_07172025.pkl", "../policies/hk_female_cb.pth", "../policies/hk_female_cb_", lr=5e-4, epochs=35)

    # Comorbidity HK
    mean, std = learn_cb_policy("../notebooks/trajectories/nonrenal_cb.pkl", "../policies/nonrenal_cb.pth", "../policies/nonrenal_cb_", lr=5e-3, epochs=40)
    mean, std = learn_cb_policy("../notebooks/trajectories/renal_cb.pkl", "../policies/renal_cb.pth", "../policies/renal_cb_", lr=1e-3, epochs=40)

    # Dosages, HK
    mean, std = learn_cb_policy("../notebooks/trajectories/low_k_cb.pkl", "../policies/low_k_cb.pth",
                                "../policies/low_k_cb_", lr=5e-3, epochs=40, task='hk')
    mean, std = learn_cb_policy("../notebooks/trajectories/high_k_cb.pkl", "../policies/high_k_cb.pth",
                                "../policies/high_k_cb_", lr=1e-3, epochs=40, task='hk')


    # Gender, HN
    mean, std = learn_cb_policy("../notebooks/trajectories/hn_male_cb.pkl", "../policies/hn_male_cb.pth", "../policies/hn_male_cb_", lr=5e-4, epochs=50, task='hn')
    mean, std = learn_cb_policy("../notebooks/trajectories/hn_female_cb.pkl", "../policies/hn_female_cb.pth", "../policies/hn_female_cb_", lr=5e-4, epochs=50, task='hn')

    # # Comorbidity, HN
    mean, std = learn_cb_policy("../notebooks/trajectories/hn_no_cirrhosis_cb.pkl", "../policies/hn_no_cirrhosis_cb.pth", "../policies/hn_no_cirrhosis_cb_", lr=5e-3, epochs=40, task='hn')
    mean, std = learn_cb_policy("../notebooks/trajectories/hn_cirrhosis_cb.pkl", "../policies/hn_cirrhosis_cb.pth", "../policies/hn_cirrhosis_cb_", lr=5e-3, epochs=40, task='hn')
    #
    mean, std = learn_cb_policy("../notebooks/trajectories/hn_low_na_cb.pkl", "../policies/hn_low_na.pth", "../policies/hn_low_na_cb_", lr=5e-3, epochs=40, task='hn')
    mean, std = learn_cb_policy("../notebooks/trajectories/hn_high_na_cb.pkl", "../policies/hn_high_na.pth", "../policies/hn_high_na_", lr=5e-3, epochs=40, task='hn')
