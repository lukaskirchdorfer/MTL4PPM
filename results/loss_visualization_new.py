# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:17:34 2025
"""
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
#import numpy as np

def main_new():
    datasets = ['BPIC20_DomesticDeclarations', 'P2P',] # 'HelpDesk']
    dataset_names = ['BPIC20DD', 'P2P']
    model = 'LSTM'
    tasks = 'NAP+NTP+RTP'
    focus_task = 'next_activity'
    mto_order = ['EW'] 

    vis_params = []
    for dataset in datasets:
        csv_name = dataset + '_best_results.csv'
        csv_path = os.path.join(os.getcwd(), dataset, csv_name)
        srch_str, task_str = comb_string(tasks)
        res_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', dataset)
        rel_loss_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_relative_loss.pdf'
        grad_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_gradient.pdf'
        loss_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_loss.pdf' 

        # Visualize validation losses
        mtl_lst, loss_lst, train_loss_lst = get_losses(
            csv_path, res_dir, srch_str, task_str, model, focus_task)
        mto_order_plus = mto_order + ['STL']
        combined = list(zip(mtl_lst, loss_lst, train_loss_lst))
        combined_sorted = sorted([item for item in combined if item[0] in mto_order_plus],
            key=lambda x: mto_order_plus.index(x[0]))
        mtl_lst, loss_lst, train_loss_lst = map(list, zip(*combined_sorted))  
        vis_params.append((mtl_lst, loss_lst, train_loss_lst, focus_task, loss_pdf))

    loss_vis_all(dataset_names, vis_params)


def main():
    dataset = 'BPIC20_DomesticDeclarations' #'Production'
    model = 'LSTM'
    tasks = 'NAP+NTP+RTP'
    focus_task = 'next_activity'
    #mto_order = ['EW', 'DWA' , 'RLW', 'UW', 'UW_SO', 'UW_O', 'GLS', 'GradDrop', 'CAGrad', 'PCGrad', 'GradNorm', 'IMTL', 'Nash_MTL']  
    mto_order = ['EW']  
    
    csv_name = dataset + '_best_results.csv'
    csv_path = os.path.join(os.getcwd(), dataset, csv_name)
    srch_str, task_str = comb_string(tasks)
    res_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', dataset)
    rel_loss_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_relative_loss.pdf'
    grad_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_gradient.pdf'
    loss_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_loss.pdf'    
    
    # # Visualize task weights, gradient cosine and magnitude similarities
    # mtl_lst, weight_lst, grad_cos_lst, grad_mag_lst = get_result(
    #     csv_path, res_dir, srch_str, task_str, model)    
    # #print(len(mtl_lst), len(weight_lst), len(grad_cos_lst), len(grad_mag_lst))    
    # combined = list(zip(mtl_lst, weight_lst, grad_cos_lst, grad_mag_lst))
    # combined_sorted = sorted([item for item in combined if item[0] in mto_order],
    #     key=lambda x: mto_order.index(x[0]))
    # mtl_lst, weight_lst, grad_cos_lst, grad_mag_lst = map(list, zip(*combined_sorted))  
    # # Visualize task weights
    # weight_vis(mtl_lst, weight_lst, focus_task, rel_loss_pdf)
    # # Visualize gradients
    # grad_vis(mtl_lst, grad_cos_lst, grad_mag_lst, grad_pdf)
    
    # Visualize validation losses
    mtl_lst, loss_lst, train_loss_lst = get_losses(
        csv_path, res_dir, srch_str, task_str, model, focus_task)
    mto_order_plus = mto_order + ['STL']
    combined = list(zip(mtl_lst, loss_lst, train_loss_lst))
    combined_sorted = sorted([item for item in combined if item[0] in mto_order_plus],
        key=lambda x: mto_order_plus.index(x[0]))
    mtl_lst, loss_lst, train_loss_lst = map(list, zip(*combined_sorted))  
    loss_vis(mtl_lst, loss_lst, train_loss_lst, focus_task, loss_pdf)
    

def grad_vis(mtl_lst, grad_cos_lst, grad_mag_lst, grad_pdf):
    
    nap_ntp_cos_lst, nap_rtp_cos_lst, ntp_rtp_cos_lst =   [], [] , []
    nap_ntp_mag_lst, nap_rtp_mag_lst, ntp_rtp_mag_lst =   [], [], []
    labels = []
    for mtl, grad_cos, grad_mag in zip(mtl_lst, grad_cos_lst, grad_mag_lst):
        nap_ntp_cos_lst.append(grad_cos['next_activity_vs_next_time'])
        nap_rtp_cos_lst.append(grad_cos['next_activity_vs_remaining_time'])
        ntp_rtp_cos_lst.append(grad_cos['next_time_vs_remaining_time'])
        nap_ntp_mag_lst.append(grad_mag['next_activity_vs_next_time'])
        nap_rtp_mag_lst.append(grad_mag['next_activity_vs_remaining_time'])
        ntp_rtp_mag_lst.append(grad_mag['next_time_vs_remaining_time'])
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
    max_epochs_nap_ntp_cos = max(len(w) for w in nap_ntp_cos_lst) 
    max_epochs_nap_rtp_cos = max(len(w) for w in nap_rtp_cos_lst)
    max_epochs_ntp_rtp_cos = max(len(w) for w in ntp_rtp_cos_lst)
    max_epochs_nap_ntp_mag = max(len(w) for w in nap_ntp_mag_lst)
    max_epochs_nap_rtp_mag = max(len(w) for w in nap_rtp_mag_lst)
    max_epochs_ntp_rtp_mag = max(len(w) for w in ntp_rtp_mag_lst)
    with PdfPages(grad_pdf) as pdf:
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))  # 2x2 grid of subplots
        for (label, grads, color) in zip(labels, nap_ntp_cos_lst, colors):
            epochs = list(range(1, len(grads) + 1))
            axes[0, 0].plot(epochs, grads, label=label, color=color)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Gradient Cosine Similarity NAP vs. NTP')
            axes[0, 0].set_ylabel('Gradient Cosine Similarity NAP vs. NTP')
            axes[0, 0].set_xticks(range(10, max_epochs_nap_ntp_cos + 1, 10))
            axes[0, 0].grid(True)
            axes[0, 0].legend()
        for (label, grads, color) in zip(labels, nap_ntp_mag_lst, colors):
            epochs = list(range(1, len(grads) + 1))
            axes[0, 1].plot(epochs, grads, label=label, color=color)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Gradient Magnitude Similarity NAP vs. NTP')
            axes[0, 1].set_ylabel('Gradient Magnitude Similarity NAP vs. NTP')
            axes[0, 1].set_xticks(range(10, max_epochs_nap_ntp_mag + 1, 10))
            axes[0, 1].grid(True)
            axes[0, 1].legend()
        for (label, grads, color) in zip(labels, nap_rtp_cos_lst, colors):
            epochs = list(range(1, len(grads) + 1))
            axes[1, 0].plot(epochs, grads, label=label, color=color)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Cosine Similarity NAP vs. RTP')
            axes[1, 0].set_ylabel('Gradient Cosine Similarity NAP vs. RTP')
            axes[1, 0].set_xticks(range(10, max_epochs_nap_rtp_cos + 1, 10))
            axes[1, 0].grid(True)
            axes[1, 0].legend()
        for (label, grads, color) in zip(labels, nap_rtp_mag_lst, colors):
            epochs = list(range(1, len(grads) + 1))
            axes[1, 1].plot(epochs, grads, label=label, color=color)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Magnitude Similarity NAP vs. RTP')
            axes[1, 1].set_ylabel('Gradient Magnitude Similarity NAP vs. RTP')
            axes[1, 1].set_xticks(range(10, max_epochs_nap_rtp_mag + 1, 10))
            axes[1, 1].grid(True)
            axes[1, 1].legend()
        for (label, grads, color) in zip(labels, ntp_rtp_cos_lst, colors):
            epochs = list(range(1, len(grads) + 1))
            axes[2, 0].plot(epochs, grads, label=label, color=color)
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Gradient Cosine Similarity NTP vs. RTP')
            axes[2, 0].set_ylabel('Gradient Cosine Similarity NTP vs. RTP')
            axes[2, 0].set_xticks(range(10, max_epochs_ntp_rtp_cos + 1, 10))
            axes[2, 0].grid(True)
            axes[2, 0].legend()
        for (label, grads, color) in zip(labels, ntp_rtp_mag_lst, colors):
            epochs = list(range(1, len(grads) + 1))
            axes[2, 1].plot(epochs, grads, label=label, color=color)
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Gradient Magnitude Similarity NTP vs. RTP')
            axes[2, 1].set_ylabel('Gradient Magnitude Similarity NTP vs. RTP')
            axes[2, 1].set_xticks(range(10, max_epochs_ntp_rtp_mag + 1, 10))
            axes[2, 1].grid(True)
            axes[2, 1].legend()
        #fig.legend(loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05))
        #handles, labels = axes[0, 0].get_legend_handles_labels()
        #fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout()
        pdf.savefig(fig)
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

def loss_vis_all(datasets, vis_params):
    """
    Plot train and validation losses for multiple datasets in one plot.
    - Different colors for datasets
    - Different line styles for val/train
    - Different markers for datasets
    - STL/MTL methods are labeled
    """
    import itertools
    
    # Assign a unique color and marker to each dataset
    color_palette = sns.color_palette("Set2", n_colors=len(datasets))
    markers = ['x', 'o', 'D', '^', 'v', 'P', '*', 'X', 'h', '+']
    
    sns.set_theme(context="talk")
    plt.figure(figsize=(10, 6))
    
    for idx, (dataset, (mtl_lst, loss_lst, train_loss_lst, focus_task, _)) in enumerate(zip(datasets, vis_params)):
        color = color_palette[idx % len(color_palette)]
        # marker = markers[idx % len(markers)]
        for mtl, val_loss, train_loss in zip(mtl_lst, loss_lst, train_loss_lst):
            # Label for legend
            if mtl == 'UW_SO':
                mtl_label = 'UW-SO'
            elif mtl == 'UW_O':
                mtl_label = 'UW-O'
            elif mtl == 'Nash_MTL':
                mtl_label = 'NashMTL'
            else:
                mtl_label = mtl
            # STL/MTL distinction
            method_type = 'STL' if mtl_label == 'STL' else 'MTL'
            # Only one legend entry per dataset/method
            # legend_label = f"{dataset} {mtl_label}" if method_type == 'STL' else f"{dataset} MTL"
            epochs = list(range(1, min(len(train_loss), len(val_loss)) + 1))
            if method_type == 'STL':
                # Plot train loss as solid line
                label = f"{dataset} STL"
                plt.plot(epochs, train_loss[:len(epochs)], label=label, color=color, linestyle='-', linewidth=5)
            else:
                label = f"{dataset} MTL"
                plt.plot(epochs, train_loss[:len(epochs)], label=label, color=color, linestyle='--', linewidth=5)
            # Plot validation loss as shaded area
            plt.fill_between(epochs, train_loss[:len(epochs)], val_loss[:len(epochs)], color=color, alpha=0.5)
    plt.xlabel("Epoch", fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel("Loss", fontsize=25)
    plt.yticks(fontsize=20)
    # plt.title("Train and Validation Losses for Multiple Datasets")
    plt.legend(fontsize=25, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("all_datasets_loss.pdf")
    plt.close()
    
    
def loss_vis(mtl_lst, loss_lst, train_loss_lst, focus_task, loss_pdf): 
    # upper level for y-axis
    #all_losses = [v for loss in losses for v in loss]
    #upper_limit = np.percentile(all_losses, 99)
    upper_limit = 10
    lower_limit = 1

    print(f"Loss list with length: {len(loss_lst)}: {loss_lst}")
    print(f"Train loss list with length: {len(train_loss_lst)}: {train_loss_lst}")
    
    labels, losses, train_losses = [], [], []
    for mtl, loss, train_loss in zip(mtl_lst, loss_lst, train_loss_lst):
        losses.append(loss)
        train_losses.append(train_loss)
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
    # plt.yscale('log')
    # plt.ylim(lower_limit, upper_limit)
    for (label, val_loss, train_loss, color) in zip(labels, losses, train_losses, colors):
        epochs_val = list(range(1, len(val_loss) + 1))
        epochs_train = list(range(1, len(train_loss) + 1))
        plt.plot(epochs_val, val_loss, label=f"{label} (Val)", color=color)
        plt.plot(epochs_train, train_loss, label=f"{label} (Train)", color=color, linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.ylabel(f"Validation Loss: {focus_task}")
    plt.xticks(range(10, max_epochs + 1, 10))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()    
    plt.savefig(loss_pdf)
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
    stl_loss, stl_train_loss = extract_losses(stl_log) 
    # get results for multi-task learning methods
    #CNN_next_activity_next_time_remaining_time_CAGrad_0.1_0.001_42.log
    df_task = df_inp[df_inp['Tasks'] == srch_str]
    df = df_task[df_task['Model'] == model].copy()
    df = df.reset_index(drop=True)
    shared_str = model + '_' + task_str + '_'    
    loss_lst, mtl_lst = [], []   
    train_loss_lst = []
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
        val_loss, train_loss = extract_losses(log_path)
        val_loss = val_loss[:-1]
        loss_lst.append(val_loss)
        train_loss_lst.append(train_loss)
    loss_lst.append(stl_loss)
    train_loss_lst.append(stl_train_loss)
    mtl_lst.append('STL')
    return mtl_lst, loss_lst, train_loss_lst
    
def extract_losses(log_path):
    val_losses = []
    train_losses = []
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split(" - ")
            if len(parts) >= 4 and "|" in parts[-1]:
                try:
                    values = parts[-1].split("|")
                    if len(values) > 4:
                        val_loss = float(values[3].strip())
                        train_loss = float(values[1].strip())
                        val_losses.append(val_loss)
                        train_losses.append(train_loss)
                except (IndexError, ValueError):
                    continue

    return val_losses, train_losses

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
    main_new() 