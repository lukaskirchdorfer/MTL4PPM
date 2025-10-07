# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:40:25 2025
@author: Keyvan Amiri Elyasi
"""
import os
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

def get_mtls(dataset):
    mtl_approaches = ['EW']
    if dataset == 'P2P':
        mtl_approaches.append('Nash_MTL')
    elif dataset == 'Production':
        mtl_approaches.append('RLW')
        mtl_approaches.append('GradNorm')
    elif dataset == 'HelpDesk':
        mtl_approaches.append('UW_O')
        mtl_approaches.append('GradDrop')
    elif dataset == 'Sepsis':
        mtl_approaches.append('CAGrad')
        mtl_approaches.append('DWA')
    elif dataset == 'BPIC20_DomesticDeclarations':   
        mtl_approaches.append('IMTL')
    elif dataset == 'BPIC20_InternationalDeclarations': 
        mtl_approaches.append('UW')
        mtl_approaches.append('Nash_MTL')
    elif dataset == 'BPI_Challenge_2012C': 
        mtl_approaches.append('GLS')
        mtl_approaches.append('GradNorm')
    elif dataset == 'BPIC15_1':  
        mtl_approaches.append('UW_O')
        mtl_approaches.append('RLW')
    elif dataset == 'BPI_Challenge_2013_incidents': 
        mtl_approaches.append('UW_O')
        mtl_approaches.append('Nash_MTL')
    return mtl_approaches
        
def main():
    datasets = ['P2P', 'HelpDesk', 'Sepsis', 'BPIC20_DomesticDeclarations', 'BPI_Challenge_2012C', 'BPI_Challenge_2013_incidents']
    task_choice = "('next_activity', 'next_time', 'remaining_time')"
    task_equivalence = 'next_activity_next_time_remaining_time'
    single_task = "('next_activity',)"
    single_equivalence = 'next_activity'
    model_choice = "CNN"
    seeds = [42, 123, 2025]
    
    root_path = os.getcwd()
    for dataset in datasets:
        best_res_name = dataset+'_best_results.csv'
        best_res_path = os.path.join(root_path, 'results', dataset, best_res_name)
        best_res = pd.read_csv(best_res_path)
        sel_model = best_res[best_res['Model']==model_choice] 
        sel_task = sel_model[sel_model['Tasks']==task_choice]              
        mtl_approaches = get_mtls(dataset)
        stl_df = sel_model[sel_model['Tasks']==single_task] 
        learning_rate = stl_df.iloc[0]['Learning Rate']
        stl_str = model_choice+'__STL_'+single_equivalence+'_task__next_activity_EW_None_'+str(learning_rate)+'_'
        model_lst = [stl_str]
        for mtl in mtl_approaches:
            mtl_df = sel_task[sel_task['MTL']==mtl]
            mtl_hpo = "None" if pd.isna(mtl_df.iloc[0]['MTL HPO']) else str(mtl_df.iloc[0]['MTL HPO'])
            learning_rate = mtl_df.iloc[0]['Learning Rate']
            mtl_str = model_choice+'__MTL_'+task_equivalence+'_task__next_activity_'+mtl+'_'+mtl_hpo+'_'+str(learning_rate)+'_'
            model_lst.append(mtl_str)
        mean_lst, std_lst = [], []
        for base_name in model_lst:
            scores = []
            for seed in seeds:
                file_name = base_name+str(seed)+'.csv'
                file_path = os.path.join(root_path, 'models', dataset, file_name)
                df = pd.read_csv(file_path)
                score = f1_score(df['ground_truth'], df['prediction'], average='macro')
                scores.append(score)
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            mean_lst.append(score_mean)
            std_lst.append(score_std)
        print(dataset)
        labels = ['STL'] + mtl_approaches
        for idx in range(len(labels)):
            print(f'{labels[idx]}: Average: {mean_lst[idx]}, Std: {std_lst[idx]}')


if __name__ == '__main__':
    main() 