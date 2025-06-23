import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict


class Trainer:
    def __init__(self, model,
                 train_dataset, val_dataset=None, test_dataset=None,
                 batch_size=32, learning_rate=0.001, device='cuda',
                 patience=10, min_delta=0.0, logger=None, **kwargs):
        """
        Trainer class for process prediction models
        
        Args:
            model (nn.Module): PyTorch model
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
            test_dataset (Dataset): Test dataset
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            device (str): Device to use for training
            **kwargs: Additional arguments
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.batch_size = batch_size
        self.kwargs = kwargs
        
        # Early Stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_epoch = 0 # To track the epoch of the best model
        
        # Store the original dataset for time denormalization
        self.original_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
            ) if val_dataset else None
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
            ) if test_dataset else None
            
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Store the active tasks (from model's task heads)
        self.active_tasks = list(self.model.task_heads.keys())
        
        # Loss functions for different tasks
        self.loss_fns = {}
        for task in self.active_tasks:
            if task == 'next_activity':
                self.loss_fns[task] = nn.CrossEntropyLoss()
            else:
                # For time-based tasks, we'll use a custom loss function
                self.loss_fns[task] = lambda pred, target, task=task: self._time_loss_fn(pred, target, task)

        self.multi_task = len(self.active_tasks) > 1

        self.loss_item = np.zeros(len(self.active_tasks))

        # Add metrics tracking
        self.metrics = {
            'next_activity': {'accuracy': 0.0},
            'next_time': {'mae_days': 0.0},
            'remaining_time': {'mae_days': 0.0}
        }
        
    def _time_loss_fn(self, pred, target, task):
        """
        Custom loss function for time-based tasks that computes MAE in days
        """
        feature_name = 'time_since_last' if task == 'next_time' else 'remaining_time'
        # Use the tensor version for differentiability
        pred_days = self.original_dataset.denormalize_time_tensor(pred, feature_name)
        target_days = self.original_dataset.denormalize_time_tensor(target, feature_name)
        return torch.mean(torch.abs(pred_days - target_days))
    
    def _compute_metrics(self, outputs, targets):
        """
        Compute metrics for each task
        
        Args:
            outputs (dict): Model outputs
            targets (dict): Ground truth targets
            
        Returns:
            dict: Dictionary of metrics for each task
        """
        metrics = {}
        
        # Compute accuracy for next activity
        if 'next_activity' in outputs and 'next_activity' in targets:
            pred = outputs['next_activity'].argmax(dim=-1)
            correct = (pred == targets['next_activity']).float().mean()
            metrics['next_activity'] = {'accuracy': correct.item()}
        
        # Compute MAE in days for time-based tasks
        for task in ['next_time', 'remaining_time']:
            if task in outputs and task in targets:
                pred_days = torch.tensor([
                    self.original_dataset.denormalize_time(
                        p.item(),
                        'time_since_last' if task == 'next_time' else 'remaining_time')
                    for p in outputs[task]
                ]).to(self.device)
                
                target_days = torch.tensor([
                    self.original_dataset.denormalize_time(
                        t.item(),
                        'time_since_last' if task == 'next_time' else 'remaining_time')
                    for t in targets[task]
                ]).to(self.device)                
                mae = torch.mean(torch.abs(pred_days - target_days))
                metrics[task] = {'mae_days': mae.item()}
        
        return metrics
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        task_losses = {task: 0 for task in self.active_tasks}
        metrics_sum = {task: {metric: 0.0 for metric in self.metrics[task].keys()} 
                      for task in self.active_tasks}
        
        for batch in tqdm(self.train_loader, desc='Training'):
            self.optimizer.zero_grad()
            
            # Move batch to device
            features = batch['features'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Forward pass
            outputs = self.model(features)
            
            # Calculate losses
            train_losses = self._compute_loss(outputs, targets)
            loss = train_losses.sum()
            
            # Update task-specific losses
            for i, task in enumerate(self.active_tasks):
                task_losses[task] += train_losses[i].item()
            
            # Compute and accumulate metrics
            batch_metrics = self._compute_metrics(outputs, targets)
            for task, task_metrics in batch_metrics.items():
                for metric, value in task_metrics.items():
                    metrics_sum[task][metric] += value

            # Backward pass
            if self.multi_task:
                task_weights = self.model.backward(train_losses, **self.kwargs)
            else:
                loss.backward()

            self.optimizer.step()
            total_loss += loss.item()
        
        # Calculate average losses and metrics
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_task_losses = {
            task: loss / num_batches
            for task, loss in task_losses.items()
        }
        
        # Calculate average metrics
        avg_metrics = {
            task: {metric: value / num_batches 
                  for metric, value in metrics_sum[task].items()}
            for task in self.active_tasks
        }
        
        for i, task in enumerate(self.active_tasks):
            self.loss_item[i] = avg_task_losses[task]
        
        return avg_loss, avg_task_losses, avg_metrics
    
    def _compute_loss(self, outputs, targets):
        """Compute losses for all active tasks"""
        train_losses = torch.zeros(len(self.active_tasks)).to(self.device)
        for i, task in enumerate(self.active_tasks):
            if task in targets and task in outputs:
                # Ensure consistent dimensions for time-based tasks
                if task != 'next_activity':
                    pred = outputs[task].view(-1)  # Flatten predictions
                    target = targets[task].view(-1)  # Flatten targets
                    # Pass task for time-based losses
                    train_losses[i] = self.loss_fns[task](pred, target, task)  
                else:
                    # Don't pass task for CrossEntropyLoss
                    train_losses[i] = self.loss_fns[task](
                        outputs[task], targets[task])  
        return train_losses
    
    def validate(self):
        """Validate the model"""
        if not self.val_loader:
            return None, None, None
            
        self.model.eval()
        total_loss = 0
        task_losses = {task: 0 for task in self.active_tasks}
        metrics_sum = {task: {metric: 0.0 for metric in self.metrics[task].keys()} 
                      for task in self.active_tasks}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                features = batch['features'].to(self.device)
                targets = {
                    k: v.to(self.device) for k, v in batch['targets'].items()}
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate losses
                val_losses = self._compute_loss(outputs, targets)
                loss = val_losses.sum()
                
                # Update task-specific losses
                for i, task in enumerate(self.active_tasks):
                    task_losses[task] += val_losses[i].item()
                
                # Compute and accumulate metrics
                batch_metrics = self._compute_metrics(outputs, targets)
                for task, task_metrics in batch_metrics.items():
                    for metric, value in task_metrics.items():
                        metrics_sum[task][metric] += value
                
                total_loss += loss.item()
        
        # Calculate average losses and metrics
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_task_losses = {
            task: loss / num_batches
            for task, loss in task_losses.items()
        }
        
        # Calculate average metrics
        avg_metrics = {
            task: {metric: value / num_batches 
                  for metric, value in metrics_sum[task].items()}
            for task in self.active_tasks
        }
        
        return avg_loss, avg_task_losses, avg_metrics
    
    def train(self, num_epochs, save_path=None):
        """Train the model for multiple epochs"""       
        # Reset early stopping state for a new training run
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_epoch = 0
        
        self.model.train_loss_buffer = np.zeros([self.model.task_num, num_epochs])
        
        # Print header
        header = "Epoch"
        for task in self.active_tasks:
            header += f" | {task.upper()} Train Loss | {task.upper()} Train {list(self.metrics[task].keys())[0].replace('_', ' ').title()}"
            if self.val_loader:
                header += f" | {task.upper()} Val Loss | {task.upper()} Val {list(self.metrics[task].keys())[0].replace('_', ' ').title()}"
        print(f'\n{header}')
        print('-' * len(header))
        self.logger.info(f'\n{header}')
        self.logger.info('-' * len(header))
        
        
        for epoch in range(num_epochs):
            self.model.epoch = epoch            
            # Train
            train_loss, train_task_losses, train_metrics = self.train_epoch()
            self.model.train_loss_buffer[:, epoch] = self.loss_item            
            # Validate
            if self.val_loader:
                val_loss, val_task_losses, val_metrics = self.validate()            
            # Print metrics in table format
            metrics_str = f"{epoch + 1:3d}"
            for task in self.active_tasks:
                metrics_str += f" | {train_task_losses[task]:.4f} | {list(train_metrics[task].values())[0]:.4f}"
                if self.val_loader:
                    metrics_str += f" | {val_task_losses[task]:.4f} | {list(val_metrics[task].values())[0]:.4f}"
            print(metrics_str)
            self.logger.info(metrics_str)
            
            # Check for early stopping only if a validation loader exists
            if self.val_loader:
                # Use val_loss for early stopping
                current_val_loss = val_loss # This is the overall validation loss
                # Check for improvement
                if current_val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = current_val_loss
                    self.epochs_no_improve = 0
                    # Store the epoch where improvement occurred
                    self.best_epoch = epoch + 1 
                    # Save the model only if it's the best so far
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    self.epochs_no_improve += 1
                    #print(f"No improvement for {self.epochs_no_improve} epochs.")
                    #self.logger.info(f"No improvement for {self.epochs_no_improve} epochs.")
                # Trigger early stopping
                if self.epochs_no_improve >= self.patience:
                    self.early_stop = True
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs "
                          f"(no improvement for {self.patience} epochs).")
                    self.logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs "
                          f"(no improvement for {self.patience} epochs).")
                    print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                    self.logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                    break # Exit the training loop
            else:
                if save_path and train_loss < self.best_val_loss:
                    self.best_val_loss = train_loss
                    torch.save(self.model.state_dict(), save_path)
    
                    
    def inference(self, inf_path=None):
        print('Now: Inference on Test set.')
        self.logger.info('Now: Inference on Test set.')
        if not self.test_loader:
            return           
        self.model.eval()
        metrics_sum = {task: {metric: 0.0 for metric in self.metrics[task].keys()} 
                      for task in self.active_tasks} 
        results = defaultdict(lambda: {'case_id': [], 'prefix_length': [],
                                       'ground_truth': [], 'prediction': []})
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Test'):
                case_ids = batch['case_id']
                prefix_lengths = batch['seq_len']
                # Move batch to device
                features = batch['features'].to(self.device)
                targets = {
                    k: v.to(self.device) for k, v in batch['targets'].items()}                
                # Forward pass
                outputs = self.model(features)
                # Compute and accumulate metrics
                batch_metrics = self._compute_metrics(outputs, targets)
                for task, task_metrics in batch_metrics.items():
                    for metric, value in task_metrics.items():
                        metrics_sum[task][metric] += value
                        
                # Get the most probable class for the classification task         
                if 'next_activity' in outputs and 'next_activity' in targets:
                    outputs['next_activity'] = outputs['next_activity'].argmax(dim=-1)  
                # Reverse normalization for regression tasks
                for task in ['next_time', 'remaining_time']:
                    if task in outputs and task in targets:
                        pred_days = torch.tensor([
                            self.original_dataset.denormalize_time(
                                p.item(),
                                'time_since_last' if task == 'next_time' else 'remaining_time')
                            for p in outputs[task]
                        ]).to(self.device)                        
                        target_days = torch.tensor([
                            self.original_dataset.denormalize_time(
                                t.item(),
                                'time_since_last' if task == 'next_time' else 'remaining_time')
                            for t in targets[task]
                        ]).to(self.device)  
                        outputs[task] = pred_days
                        targets[task] = target_days                    
                for task in self.active_tasks:
                    gt = targets[task].detach().cpu().numpy()
                    pred = outputs[task].detach().cpu().numpy()
                    # Handle shape for regression (squeeze 1D)
                    if pred.ndim == 2 and pred.shape[1] == 1:
                        pred = pred.squeeze(-1)
                    if gt.ndim == 2 and gt.shape[1] == 1:
                        gt = gt.squeeze(-1)
                    # Append batch results
                    results[task]['case_id'].extend(case_ids)
                    results[task]['prefix_length'].extend(prefix_lengths)
                    results[task]['ground_truth'].extend(gt.tolist())
                    results[task]['prediction'].extend(pred.tolist())
                    
        # convert inference results to dataframes
        task_counter = 0
        for task in self.active_tasks:
            current_df = pd.DataFrame({
                'case_id': results[task]['case_id'],
                'prefix_length': results[task]['prefix_length'],
                'ground_truth': results[task]['ground_truth'],
                'prediction': results[task]['prediction']}) 
            current_df['prefix_length'] = current_df[
                'prefix_length'].apply(lambda x: x.item())
            current_df.to_csv(inf_path[task_counter], index=False)
            task_counter += 1        
        
        # Calculate average metrics
        num_batches = len(self.test_loader)      
        avg_metrics = {
            task: {metric: value / num_batches 
                  for metric, value in metrics_sum[task].items()}
            for task in self.active_tasks
        }  
        metrics_str = ''
        header = ''
        for task in self.active_tasks:
            if task == 'next_activity':
                header += f" | {task.upper()}   Accuracy"
            else:
                header += f" | {task.upper()}   MAE(Days)"
            metrics_str += f" | {list(avg_metrics[task].values())[0]:.4f}"
        print(header)
        print(metrics_str)
        self.logger.info(header)
        self.logger.info(metrics_str)