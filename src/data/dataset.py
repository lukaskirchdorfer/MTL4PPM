import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

class ProcessLogDataset(Dataset):
    def __init__(self, csv_path, tasks=None):
        """
        Dataset class for process logs
        
        Args:
            csv_path (str): Path to the CSV file
            max_len (int): Maximum sequence length
            tasks (list): List of tasks to perform. Default is ['next_activity']
        """
        self.tasks = tasks if tasks is not None else ['next_activity']
        
        # Load and preprocess data
        self.df = pd.read_csv(csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        # add days and hours of the last event
        self.df['day'] = pd.to_datetime(self.df['timestamp']).dt.day_name()
        self.df['hour'] = (pd.to_datetime(self.df['timestamp']).dt.hour + 
                           pd.to_datetime(self.df['timestamp']).dt.minute / 60)
        self.df['hour'] = ((self.df['hour'] - self.df['hour'].min()) / 
                           (self.df['hour'].max() - self.df['hour'].min()))
        
        # Encode categorical variables
        self.activity_encoder = LabelEncoder()
        self.resource_encoder = LabelEncoder()
        self.day_encoder = LabelEncoder()
        self.df['activity_encoded'] = self.activity_encoder.fit_transform(self.df['activity'])
        self.df['resource_encoded'] = self.resource_encoder.fit_transform(self.df['resource'])
        self.df['day_encoded'] = self.day_encoder.fit_transform(self.df['day'])
        
        # Group by case_id
        self.cases = self.df.groupby('case_id')
        self.case_ids = list(self.cases.groups.keys())
        
        # Store normalization parameters
        self.time_means = {}
        self.time_stds = {}

        self.max_len = self._get_max_case_length()
        print(f"Max length: {self.max_len}")
        
        # Calculate time-based features
        self._calculate_time_features()
        
        # Create all training examples
        self.examples = self._create_examples()

    def _get_max_case_length(self):
        return max(len(self.cases.get_group(case_id)) for case_id in self.case_ids)
        
    def _calculate_time_features(self):
        """Calculate time-based features for each event"""
        # Calculate time since start of case (in seconds)
        self.df['time_since_start'] = self.df.groupby('case_id')['timestamp'].transform(
            lambda x: (x - x.iloc[0]).dt.total_seconds())
        
        # Calculate time since last event (in seconds)
        self.df['time_since_last'] = self.df.groupby('case_id')['timestamp'].transform(
            lambda x: x.diff().dt.total_seconds()).fillna(0)
        
        # Calculate remaining time (in seconds)
        self.df['remaining_time'] = self.df.groupby('case_id')['timestamp'].transform(
            lambda x: (x.iloc[-1] - x).dt.total_seconds())
        
        # Normalize time-based features if they are needed
        if any(task in ['next_time', 'remaining_time'] for task in self.tasks):
            # Convert seconds to days for better interpretability
            for col in ['time_since_start', 'time_since_last', 'remaining_time']:
                self.df[col] = self.df[col] / (24 * 3600)
                
                # Log-transform and normalize time features
                self.df[col] = np.log1p(self.df[col])
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df[col] = (self.df[col] - mean) / (std + 1e-7)
                
                # Store normalization parameters
                self.time_means[col] = mean
                self.time_stds[col] = std
    
    def denormalize_time(self, normalized_value, feature_name):
        """
        Convert normalized time value back to days
        
        Args:
            normalized_value (float): Normalized time value
            feature_name (str): Name of the time feature ('time_since_start', 'time_since_last', or 'remaining_time')
            
        Returns:
            float: Time value in days
        """
        if feature_name not in self.time_means:
            raise ValueError(f"Unknown time feature: {feature_name}")
            
        # Reverse normalization
        denormalized = normalized_value * self.time_stds[feature_name] + self.time_means[feature_name]
        # Reverse log transformation
        return np.expm1(denormalized)
    
    def _create_examples(self):
        """Create training examples for each event in each case"""
        examples = []
        for case_id in self.case_ids:
            case_data = self.cases.get_group(case_id)
            case_data = case_data.sort_values(by='timestamp')
            
            # For each event in the case (except the last one)
            for i in range(len(case_data) - 1):
                # Get the prefix of events up to the current point
                prefix = case_data.iloc[:i+1]
                
                # Create feature matrix for the prefix
                features = np.column_stack([
                    prefix['activity_encoded'].values,
                    prefix['resource_encoded'].values,
                    prefix['day_encoded'].values,
                    prefix['hour'].values,
                    prefix['time_since_last'].values,
                    prefix['time_since_start'].values
                ])
                
                # Pad the beginning of the sequence with zeros
                if len(features) < self.max_len:
                    pad_length = self.max_len - len(features)
                    features = np.pad(features, ((pad_length, 0), (0, 0)), mode='constant')
                
                # Get the target event (next event)
                target_event = case_data.iloc[i+1]
                
                # Create target dictionary
                targets = {}
                if 'next_activity' in self.tasks:
                    targets['next_activity'] = target_event['activity_encoded']
                if 'next_time' in self.tasks:
                    targets['next_time'] = target_event['time_since_last']
                if 'remaining_time' in self.tasks:
                    targets['remaining_time'] = target_event['remaining_time']
                
                # collect examples
                examples.append({
                    'features': features,
                    'targets': targets,
                    'seq_len': len(prefix),
                    'case_id': case_id
                })                
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert features to tensor
        features = torch.FloatTensor(example['features'])
        
        # Convert targets to tensors
        targets = {}
        for task, value in example['targets'].items():
            if task == 'next_activity':
                targets[task] = torch.tensor(value, dtype=torch.long)
            else:
                targets[task] = torch.tensor([value], dtype=torch.float32)
        
        return {
            'features': features,
            'targets': targets,
            'seq_len': example['seq_len'],
            'case_id': example['case_id']
        }
    
    @property
    def num_activities(self):
        return len(self.activity_encoder.classes_)
    
    @property
    def num_resources(self):
        return len(self.resource_encoder.classes_)
    
    @property
    def num_days(self):
        return len(self.day_encoder.classes_)
    
    @property
    def feature_dim(self):
        return 6  # activity, resource, day, hour, time since last, time since start 
    
    @property
    def num_feat_dim(self):
        return 3  # hour, time since last, time since start 