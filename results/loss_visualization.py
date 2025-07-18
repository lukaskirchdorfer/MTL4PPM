# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:17:34 2025
@author: kamirel
"""
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    dataset = 'Production'
    model = 'CNN'
    tasks = 'NAP+NTP+RTP'
    focus_task = 'next_activity'
    #mto_order = ['EW', 'DWA' , 'RLW', 'UW', 'UW_SO', 'UW_O', 'GLS', 'GradDrop', 'CAGrad', 'PCGrad', 'GradNorm', 'IMTL', 'Nash_MTL']  
    mto_order = ['EW', 'RLW', 'UW', 'PCGrad', 'GradNorm', 'IMTL']  
    
    csv_name = dataset + '_best_results.csv'
    csv_path = os.path.join(os.getcwd(), dataset, csv_name)
    srch_str, task_str = comb_string(tasks)
    res_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', dataset)
    rel_loss_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_relative_loss.pdf'
    loss_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_loss.pdf'    
    
    mtl_lst, weight_lst, grad_cos_lst, grad_mag_lst = get_result(
        csv_path, res_dir, srch_str, task_str, model)    
    #print(len(mtl_lst), len(weight_lst), len(grad_cos_lst), len(grad_mag_lst))    
    combined = list(zip(mtl_lst, weight_lst, grad_cos_lst, grad_mag_lst))
    combined_sorted = sorted([item for item in combined if item[0] in mto_order],
        key=lambda x: mto_order.index(x[0]))
    mtl_lst, weight_lst, grad_cos_lst, grad_mag_lst = map(list, zip(*combined_sorted))    
    weight_vis(mtl_lst, weight_lst, focus_task, rel_loss_pdf)
    
    mtl_lst, loss_lst = get_losses(
        csv_path, res_dir, srch_str, task_str, model, focus_task)
    mto_order_plus = mto_order + ['STL']
    combined = list(zip(mtl_lst, loss_lst))
    combined_sorted = sorted([item for item in combined if item[0] in mto_order_plus],
        key=lambda x: mto_order_plus.index(x[0]))
    mtl_lst, loss_lst = map(list, zip(*combined_sorted))  
    loss_vis(mtl_lst, loss_lst, focus_task, loss_pdf)

def loss_vis(mtl_lst, loss_lst, focus_task, loss_pdf): 
    # upper level for y-axis
    #all_losses = [v for loss in losses for v in loss]
    #upper_limit = np.percentile(all_losses, 99)
    upper_limit = 10
    lower_limit = 1
    
    labels, losses = [], []
    for mtl, loss in zip(mtl_lst, loss_lst):
        losses.append(loss)
        if mtl == 'UW_SO':
            labels.append('UW-SO')
        elif mtl == 'UW_O':
            labels.append('UW-O')
        elif mtl == 'Nash_MTL':
            labels.append('NashMTL')
        else:
            labels.append(mtl)   
            

    if len(mtl_lst) > 9:
        colors = sns.color_palette("tab20", n_colors=len(labels))
    else:
        colors = sns.color_palette("tab10", n_colors=len(labels))
    sns.set_theme(context="talk")
    max_epochs = max(len(w) for w in losses)
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.ylim(lower_limit, upper_limit)
    for (label, weights, color) in zip(labels, losses, colors):
        epochs = list(range(1, len(weights) + 1))
        plt.plot(epochs, weights, label=label, color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    #plt.ylabel(f"Validation Loss: {focus_task}")
    plt.xticks(range(10, max_epochs + 1, 10))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()    
    plt.savefig(loss_pdf)
    plt.close()
    
def weight_vis(mtl_lst, weight_lst, focus_task, rel_loss_pdf):    
    labels, focus_weights = [], []
    for mtl, task_weights in zip(mtl_lst, weight_lst):
        num_epochs = len(next(iter(task_weights.values())))
        task_names = list(task_weights.keys())
        for epoch in range(num_epochs):
            total = sum(task_weights[task][epoch] for task in task_names)
            task_weights[focus_task][epoch] /= total
        values = [w.item() if isinstance(w, torch.Tensor) else w for w in task_weights[focus_task]]
        focus_weights.append(values)
        if mtl == 'UW_SO':
            labels.append('UW-SO')
        elif mtl == 'UW_O':
            labels.append('UW-O')
        elif mtl == 'Nash_MTL':
            labels.append('NashMTL')
        else:
            labels.append(mtl)         
    if len(mtl_lst) > 10:
        colors = sns.color_palette("tab20", n_colors=len(labels))
    else:
        colors = sns.color_palette("tab10", n_colors=len(labels))
    sns.set_theme(context="talk")
    max_epochs = max(len(w) for w in focus_weights)
    plt.figure(figsize=(10, 6))
    for (label, weights, color) in zip(labels, focus_weights, colors):
        epochs = list(range(1, len(weights) + 1))
        plt.plot(epochs, weights, label=label, color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Relatvie Loss Weight")
    #plt.ylabel(f"Relatvie Loss Weight: {focus_task}")
    plt.xticks(range(10, max_epochs + 1, 10))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()    
    plt.savefig(rel_loss_pdf)
    plt.close()
    
          
def get_result(csv_path, res_dir, srch_str, task_str, model, seed=42):
    # get all relevant MTL results (loss weights, gradient magnitude and cosine similarity)
    df_inp = pd.read_csv(csv_path)
    df_task = df_inp[df_inp['Tasks'] == srch_str]
    df = df_task[df_task['Model'] == model].copy()
    df = df.reset_index(drop=True)
    shared_str = model + '_' + task_str + '_'
    weight_lst, grad_cos_lst, grad_mag_lst, mtl_lst = [], [], [], []
    for i in range (len(df)):
        mtl = df.loc[i, 'MTL'] 
        mtl_lst.append(mtl)
        lr = df.loc[i, 'Learning Rate']
        if pd.isna(df.loc[i, 'MTL HPO']):
            hpo = 'None'
        else:
            hpo = str(df.loc[i, 'MTL HPO'])
        base = shared_str + mtl + '_' + hpo + '_' + str(lr) + '_' + str(seed)
        weight_path = os.path.join(res_dir, base + '_task_weights.json')
        grad_mag_path = os.path.join(res_dir, base + '_gradient_magnitude.pt')
        grad_cos_path = os.path.join(res_dir, base + '_gradient_cosine.pt')
        task_weights = json.load(open(weight_path))
        grad_cos_sim = torch.load(grad_cos_path)
        grad_mag_sim = torch.load(grad_mag_path)
        weight_lst.append(task_weights)
        grad_cos_lst.append(grad_cos_sim)
        grad_mag_lst.append(grad_mag_sim)
    return mtl_lst, weight_lst, grad_cos_lst, grad_mag_lst

def get_losses(csv_path, res_dir, srch_str, task_str, model, focus_task, seed=42):
    df_inp = pd.read_csv(csv_path)    
    # get results for single task learning    
    stl_str = f"('{focus_task}',)"
    stl_df = df_inp[(df_inp['Model'] == model) & (df_inp['Tasks'] == stl_str)].copy()
    stl_df = stl_df.reset_index(drop=True)
    lr = stl_df.loc[0, 'Learning Rate']
    stl_log = os.path.join(
        res_dir, model + '_' + focus_task + '_EW_None_' + str(lr) + '_' + str(seed) + '.log')
    stl_loss = extract_losses(stl_log) 
    # get results for multi-task learning methods
    #CNN_next_activity_next_time_remaining_time_CAGrad_0.1_0.001_42.log
    df_task = df_inp[df_inp['Tasks'] == srch_str]
    df = df_task[df_task['Model'] == model].copy()
    df = df.reset_index(drop=True)
    shared_str = model + '_' + task_str + '_'    
    loss_lst, mtl_lst = [], []    
    for i in range (len(df)):
        mtl = df.loc[i, 'MTL'] 
        mtl_lst.append(mtl)
        lr = df.loc[i, 'Learning Rate']
        if pd.isna(df.loc[i, 'MTL HPO']):
            hpo = 'None'
        else:
            hpo = str(df.loc[i, 'MTL HPO'])
        log_name = shared_str + mtl + '_' + hpo + '_' + str(lr) + '_' + str(seed) + '.log'
        log_path = os.path.join(res_dir, log_name)
        val_loss = extract_losses(log_path)
        val_loss = val_loss[:-1]
        loss_lst.append(val_loss)
    loss_lst.append(stl_loss)
    mtl_lst.append('STL')
    return mtl_lst, loss_lst
    
def extract_losses(log_path):
    val_losses = []
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split(" - ")
            if len(parts) >= 4 and "|" in parts[-1]:
                try:
                    values = parts[-1].split("|")
                    val_loss = float(values[3].strip())
                    val_losses.append(val_loss)
                except (IndexError, ValueError):
                    continue
    return val_losses

def comb_string(combination):
    if combination == 'NAP+NTP+RTP':
        str1 = "('next_activity', 'next_time', 'remaining_time')"
        str2 = 'next_activity_next_time_remaining_time'
    elif combination == 'NTP+RTP':
        str1 = "('next_time', 'remaining_time')"
        str2 = 'next_time_remaining_time'
    elif combination == 'NAP+RTP':
        str1 = "('next_activity', 'remaining_time')"
        str2 = 'next_activity_remaining_time'
    elif combination == 'NAP+NTP':
        str1 = "('next_activity', 'next_time')"
        str2 = 'next_activity_next_time'
    else:
        str1 = 'All_Combinations'
        str2 = None
    return str1, str2

if __name__ == '__main__':
    main() 