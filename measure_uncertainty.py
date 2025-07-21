import torch
from src.data.dataset import ProcessLogDataset
import os
from src.models.models import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import src.weighting as weighting_method
import numpy as np

DEVICE = 'cpu'
DATASET = 'BPIC20_InternationalDeclarations'
TASKS = ['next_activity', 'next_time', 'remaining_time']
#TASKS = ['next_activity']
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2
SEED = 42
MODEL = 'CNN'
MTO = 'EW'
LR = 0.001

# Create the full dataset first
full_dataset = ProcessLogDataset(
    csv_path=os.path.join(os.getcwd(), "data", f"{DATASET}.csv"),
    tasks=TASKS  
)

# Split into train and validation sets
train_val_size = int((1 - TEST_SPLIT) * len(full_dataset))
train_size = int((1 - VAL_SPLIT) * train_val_size)    
val_size = train_val_size - train_size
test_size = len(full_dataset) - train_val_size

# Create generator for reproducible splitting
generator = torch.Generator()
generator.manual_seed(SEED)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size], generator=generator)
print(f'Train size: {len(train_dataset)}')
print(f'Validation size: {len(val_dataset)}')
print(f'Test size: {len(test_dataset)}')

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) if test_dataset else None

output_dims = {}
if len(TASKS) > 1:  # Multi-task case
    if 'next_activity' in TASKS:
        output_dims['next_activity'] = full_dataset.num_activities
    if 'next_time' in TASKS:
        output_dims['next_time'] = 1
    if 'remaining_time' in TASKS:
        output_dims['remaining_time'] = 1
else:  # Single task case
    if TASKS[0] != 'next_activity':
        output_dims = {TASKS[0]: 1}
    else:
        output_dims = {'next_activity': full_dataset.num_activities}

if MODEL == 'LSTM':
    model_parameters = {
        "input_dim": full_dataset.feature_dim,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "num_activities": full_dataset.num_activities,
    }
elif MODEL == 'CNN':
    model_parameters = {
        "input_dim": full_dataset.feature_dim,
        "hidden_dim": 128,
        "num_filters": 64,
        "kernel_size": 3,
        "dropout": 0.1,
        "num_activities": full_dataset.num_activities,
        "max_len": full_dataset.max_len,
    }
elif MODEL == 'Transformer':
    model_parameters = {
        "num_feat_dim": full_dataset.num_feat_dim,
        "emb_dim": 61,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.1,
        "num_activities": full_dataset.num_activities,
        "num_resources": full_dataset.num_resources, 
        "num_days": full_dataset.num_days,
        "pooling": 'mean',
        "max_len": full_dataset.max_len,
    }
weighting = weighting_method.__dict__[MTO]

# load pretrained model
model = get_model(
        model_name=MODEL,
        weighting=weighting,
        output_dims=output_dims,
        model_parameters=model_parameters,
        device=DEVICE,
        rep_grad=False,
    )
task_str = '_'.join(TASKS)
PATH_TO_MODEL = os.path.join(os.getcwd(), "models", DATASET, f"{MODEL}_{task_str}_{MTO}_None_{LR}_{SEED}_model.pt")
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=torch.device('cpu')))

# measure uncertainty
model.eval()
entropy_list = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Test'):
        case_ids = batch['case_id']
        prefix_lengths = batch['seq_len']
        # Move batch to device
        features = batch['features'].to(DEVICE)
        targets = {
            k: v.to(DEVICE) for k, v in batch['targets'].items()}                
        # Forward pass
        outputs = model(features)

        logits = outputs['next_activity']
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        entropy_list.append(entropy)

entropy_list = torch.cat(entropy_list)
print(f"Entropy: {entropy_list.mean()}")
print(f"Maximum possible entropy: {np.log(full_dataset.num_activities)}")
print(f"Entropy ratio: {entropy_list.mean() / np.log(full_dataset.num_activities)}")

