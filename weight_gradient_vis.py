# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 18:45:25 2025

@author: kamirel
"""
import os
import argparse
import json
import torch

def main():   
    parser = argparse.ArgumentParser(
        description='Visualization for task weights and gradients')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, default='LSTM',
                      choices=['LSTM', 'CNN', 'Transformer'],
                      help='Model architecture to use')
    parser.add_argument('--tasks', type=str, nargs='+', default=['next_activity'],
                      choices=['next_activity', 'next_time', 'remaining_time', 'multi'],
                      help='Prediction tasks. Use "multi" for all tasks or specify individual tasks')
    parser.add_argument('--weighting', type=str, default='EW',
                      choices=['EW', 'UW', 'DWA', 'GLS', 'IMTL', 'RLW',
                               'CAGrad', 'GradNorm', 'GradDrop', 'PCGrad',
                               'Nash_MTL', 'UW_SO', 'Scalarization'],
                      help='Weighting strategy to use')
    args = parser.parse_args()  
    
    if args.weighting == 'UW_SO' and not args.use_softmax:
        mtl_name = 'UW_O'
    else:
        mtl_name = args.weighting  
    # If 'multi' is specified, use all tasks
    tasks = ['next_activity', 'next_time', 'remaining_time'] if 'multi' in args.tasks else args.tasks
    root_path = os.getcwd()
    save_dir = os.path.join(root_path, 'models', args.dataset)
    task_weight_path = os.path.join(
        save_dir,
        f'{args.model}_{"_".join(tasks)}_{mtl_name}_task_weights.json')
    gradient_cosine_path = os.path.join(
        save_dir,
        f'{args.model}_{"_".join(tasks)}_{mtl_name}_gradient_cosine.pt')
    gradient_magnitude_path = os.path.join(
        save_dir,
        f'{args.model}_{"_".join(tasks)}_{mtl_name}_gradient_magnitude.pt')
    
    task_weights = json.load(open(task_weight_path))
    gradient_cosine_sim = torch.load(gradient_cosine_path)
    gradient_magnitude_sim = torch.load(gradient_magnitude_path)
    print(task_weights)
    print(gradient_cosine_sim)
    print(gradient_magnitude_sim)

if __name__ == '__main__':
    main() 
