# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 10:07:22 2025
@author: Keyvan Amiri Elyasi
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def angle_to_cosine_similarity(angle, degrees=True):
    if degrees:
        angle = np.radians(angle)    
    return np.cos(angle)

def cosine_similarity_to_angle(cosine, degrees=True):
    angle = np.arccos(cosine)
    return np.degrees(angle) if degrees else angle

def get_gradient_names(dataset):
    if dataset == 'Production':
        cosine_name = 'CNN_next_activity_next_time_remaining_time_EW_None_0.001_42_gradient_cosine.pt'
        magnitude_name = 'CNN_next_activity_next_time_remaining_time_EW_None_0.001_42_gradient_magnitude.pt'  
    else:
        cosine_name = 'CNN_next_activity_next_time_remaining_time_EW_None_0.0001_42_gradient_cosine.pt'
        magnitude_name = 'CNN_next_activity_next_time_remaining_time_EW_None_0.0001_42_gradient_magnitude.pt'
    return cosine_name, magnitude_name


 
def plot_to_pdf(cosine_vals_list, grad_vals_list, titles, 
                angle=100, ratio=5, pdf_path=None):
    thresh1=angle_to_cosine_similarity(angle)
    thresh2= 1/ratio
    print(thresh1, thresh2)
    plt.rcParams.update({
        'font.size': 16,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'legend.fontsize': 16,
        'legend.title_fontsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=False)  # Slightly taller for better spacing
    for i, ax in enumerate(axes):
        cos, grad = np.array(cosine_vals_list[i]), np.array(grad_vals_list[i])
        x = np.arange(1, len(cos)+1)
        ax.plot(x, cos, color="cyan", linewidth=2)
        ax.plot(x, grad, color="olive", linestyle="--", linewidth=2)        
        ax.fill_between(x, 0, 1, where=(cos<thresh1)&(grad<thresh2), 
                       color="red", alpha=0.2, transform=ax.get_xaxis_transform())        
        ax.set_ylabel("Similarity", fontweight='bold', fontsize=18)
        ax.set_title(titles[i], fontweight='bold', fontsize=18, pad=15)        
        # Bold axis ticks and labels
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.set_xlabel("Epoch", fontweight='bold', fontsize=18)        
        # Make spines thicker for better visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)    
    fig.legend(["Cosine Similarity", "Magnitude Similarity"], 
               loc="upper center", 
               ncol=2,
               fontsize=18,
               frameon=True,
               fancybox=True,
               shadow=True,
               framealpha=0.9)    
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjusted for legend space
    plt.savefig(pdf_path, format="pdf", dpi=600, bbox_inches='tight')
    plt.close()

def main():
    datasets = ['P2P', 'HelpDesk', 'Sepsis', 'BPIC20_DomesticDeclarations', 'BPI_Challenge_2012C', 'BPI_Challenge_2013_incidents']
    task_choice = "('next_activity', 'next_time', 'remaining_time')"
    task_equivalence = 'next_activity_next_time_remaining_time'
    model_choice = "CNN"
    keys = ['next_activity_vs_next_time', 'next_activity_vs_remaining_time',
            'next_time_vs_remaining_time']
    seeds = [42, 123, 2025]
    cosine_threshold = 0.0
    magnitude_threshold = 0.5 #e.g., 0.5 means one gradient is twice the other
    #titles = ["NAP vs. NTP", "NAP vs. RTP", "NTP vs. RTP"]   
    
    root_path = os.getcwd()
    log_names, mto_names, key_names, props, cos_means, mag_means= [], [], [], [], [], []
    for dataset in datasets:
        best_res_name = dataset+'_best_results.csv'
        best_res_path = os.path.join(root_path, 'results', dataset, best_res_name)
        best_res = pd.read_csv(best_res_path)
        #print(len(best_res))
        sel_task = best_res[best_res['Tasks']==task_choice]
        sel_model = sel_task[sel_task['Model']==model_choice]
        #print(len(sel_model)) 
        for _, row in sel_model.iterrows():
            mtl_name = row['MTL']
            mtl_hpo = "None" if pd.isna(row['MTL HPO']) else str(row['MTL HPO'])
            learning_rate = row['Learning Rate']
            base_str = model_choice+'_'+task_equivalence+'_'+mtl_name+'_'+str(mtl_hpo)+'_'+str(learning_rate)
            for key in keys:
                log_names.append(dataset)
                mto_names.append(mtl_name)
                key_names.append(key)
                total_prop = 0
                total_angles = 0
                total_magnitude = 0
                for seed in seeds:
                    cosine_name = base_str+'_'+str(seed)+'_gradient_cosine.pt'
                    magni_name = base_str+'_'+str(seed)+'_gradient_magnitude.pt'
                    cosine_path = os.path.join(root_path, 'models', dataset, cosine_name)
                    magni_path = os.path.join(root_path, 'models', dataset, magni_name)
                    cosine_grads = torch.load(cosine_path)[key]
                    magnitude_tensors = torch.load(magni_path)[key]
                    magnitude_grads = [t.item() for t in magnitude_tensors]
                    prop = sum(c < cosine_threshold and m < magnitude_threshold
                               for c, m in zip(cosine_grads, magnitude_grads)
                               ) / len(cosine_grads)
                    total_prop += prop
                    angles = [cosine_similarity_to_angle(c) for c in cosine_grads]
                    mean_degree = sum(angles) / len(angles)
                    total_angles += mean_degree
                    magnitudes = [1/c for c in magnitude_grads]
                    mean_magnitude = sum(magnitudes) / len(magnitudes)
                    total_magnitude += mean_magnitude
                mean_prop = total_prop/len(seeds)
                cosine_mean = total_angles/len(seeds)
                magnitude_mean = total_magnitude/len(seeds)
                props.append(mean_prop)
                cos_means.append(cosine_mean)
                mag_means.append(magnitude_mean)
        result_dict = {'dataset': log_names, 'MTL': mto_names,
                       'task_pair': key_names, 'risk_prop': props,
                       'Average_degree': cos_means, 'average_magnitude': mag_means} 
        df = pd.DataFrame.from_dict(result_dict)
        csv_path = os.path.join(root_path, 'negative_transfer.csv')
        df.to_csv(csv_path, index=False)
   
    """
    for dataset in datasets:
        pdf_name = dataset+'_Gradients.pdf'
        cosine_name, magnitude_name = get_gradient_names(dataset)
        plot_path = os.path.join(root_path, pdf_name)    
        cosine_path = os.path.join(root_path, 'models', dataset, cosine_name)   
        magnitude_path = os.path.join(root_path, 'models', dataset, magnitude_name)
        cosine_vals_list, grad_vals_list = [], []
        for key in keys:    
            cosine_grads = torch.load(cosine_path)[key]
            cosine_vals_list.append(cosine_grads)
            magnitude_tensors = torch.load(magnitude_path)[key]
            magnitude_grads = [t.item() for t in magnitude_tensors]
            grad_vals_list.append(magnitude_grads) 
        plot_to_pdf(cosine_vals_list, grad_vals_list, titles, pdf_path=plot_path)
    """

if __name__ == '__main__':
    main() 