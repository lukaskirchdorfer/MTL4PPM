# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 18:45:25 2025

@author: kamirel
"""
import os
import argparse
import json
import torch
import matplotlib.pyplot as plt
import math

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
    task_weight_plot_path = os.path.join(
        save_dir,
        f'{args.model}_{"_".join(tasks)}_{mtl_name}_task_weights.pdf.pdf')
    gradient_cosine_plot_path = os.path.join(
        save_dir,
        f'{args.model}_{"_".join(tasks)}_{mtl_name}_gradient_cosine_similarity.pdf')
    gradient_magnitude_plot_path = os.path.join(
        save_dir,
        f'{args.model}_{"_".join(tasks)}_{mtl_name}_gradient_magnitude_similarity.pdf')
    
    task_weights = json.load(open(task_weight_path))
    gradient_cosine_sim = torch.load(gradient_cosine_path)
    gradient_magnitude_sim = torch.load(gradient_magnitude_path)
    
    # Visualize task_weights
    num_epochs = len(next(iter(task_weights.values())))
    task_names = list(task_weights.keys())
    for epoch in range(num_epochs):
        total = sum(task_weights[task][epoch] for task in task_names)
        for task in task_names:
            task_weights[task][epoch] /= total
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(10, 6))
    for task in task_names:
        values = [w.item() if isinstance(w, torch.Tensor) else w for w in task_weights[task]]
        plt.plot(epochs, values, label=task)
    plt.xlabel("Epoch")
    plt.ylabel("Relative Weight")
    plt.title(
        f"Relative Task Weights: {args.dataset}- {args.model}- {args.weighting}",
        fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(task_weight_plot_path)
    plt.close()
    
    # Visualize gradient cosine similarity
    all_values = [v.item() if isinstance(v, torch.Tensor) else v 
                  for values in gradient_cosine_sim.values() for v in values]
    y_min = min(all_values)
    y_max = max(all_values)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Gradient Cosine Similarity: {args.dataset} - {args.model} - {args.weighting}",
        fontsize=16)
    for key, values in gradient_cosine_sim.items():
        epochs = list(range(1, len(values) + 1))
        sims = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        ax.plot(epochs, sims, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Cosine Similarity")
    margin = 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.legend()
    plt.tight_layout()
    plt.savefig(gradient_cosine_plot_path)
    plt.close()
    
    # Visualize gradient magnitude similarity
    all_values = [v.item() if isinstance(v, torch.Tensor) else v 
                  for values in gradient_magnitude_sim.values() for v in values]
    y_min = min(all_values)
    y_max = max(all_values)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Gradient Magnitude Similarity: {args.dataset} - {args.model} - {args.weighting}",
        fontsize=16)
    for key, values in gradient_magnitude_sim.items():
        epochs = list(range(1, len(values) + 1))
        sims = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        ax.plot(epochs, sims, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Magnitude Similarity")
    margin = 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.legend()
    plt.tight_layout()
    plt.savefig(gradient_magnitude_plot_path)
    plt.close()
    
if __name__ == '__main__':
    main() 
