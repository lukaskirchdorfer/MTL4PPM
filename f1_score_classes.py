# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:06:32 2025
@author: Keyvan Amiri Elyasi
"""
import os
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_mtl(dataset):
    if dataset in {'P2P', 'Production', 'BPIC20_DomesticDeclarations'}:
        mtl = 'EW'
    elif dataset in {'HelpDesk', 'BPIC15_1', 'BPI_Challenge_2013_incidents'}:
        mtl = 'UW_O'
    elif dataset == 'Sepsis':
        mtl = 'CAGrad'
    elif dataset == 'BPIC20_InternationalDeclarations': 
        mtl = 'UW'
    elif dataset == 'BPI_Challenge_2012C': 
        mtl = 'GLS'
    return mtl


def compare_model_f1_bar(mtl_df, stl_df, mtl, save_path):
    sns.set_theme(context="talk")
    # mtl_label = 'MTL('+mtl+')'
    mtl_label = 'MTL'
    # Get unique classes (based on ground truth)
    classes = sorted(mtl_df['ground_truth'].unique())
    # Compute per-class F1 for both models
    mtl_f1 = f1_score(mtl_df['ground_truth'], mtl_df['prediction'], average=None, labels=classes)
    stl_f1 = f1_score(stl_df['ground_truth'], stl_df['prediction'], average=None, labels=classes)
    # Compute class frequencies
    class_freq = mtl_df['ground_truth'].value_counts().reindex(classes)
    # Build DataFrame
    results = pd.DataFrame({
        'class': classes,
        'frequency': class_freq.values,
        'MTL_F1': mtl_f1,
        'STL_F1': stl_f1
    })
    # Exclude classes where both models have F1 = 0
    results = results[~((results['MTL_F1'] == 0) & (results['STL_F1'] == 0))]
    # Sort by frequency (descending)
    results = results.sort_values(by='frequency', ascending=False)
    # Plot
    plt.figure(figsize=(10, 5))
    width = 0.35
    x = range(len(results))
    plt.bar([i - width/2 for i in x], results['STL_F1'], width, label='STL', alpha=0.7)
    plt.bar([i + width/2 for i in x], results['MTL_F1'], width, label=mtl_label, alpha=0.7)
    plt.yticks(fontsize=20)
    plt.xticks(x, results['class'], rotation=45, ha='right', fontsize=20)
    plt.ylabel('F1-score', fontsize=24)
    plt.xlabel('Class (sorted by frequency)', fontsize=24)
    #plt.title('Per-Class F1-score Comparison (MTL vs. STL)')
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=600, bbox_inches='tight') 
    plt.close()


def main():
    dataset = 'BPI_Challenge_2013_incidents' # 'Production' 'BPIC20_InternationalDeclarations'
    seed = 42
    task_choice = "('next_activity', 'next_time', 'remaining_time')"
    task_equivalence = 'next_activity_next_time_remaining_time'
    single_task = "('next_activity',)"
    single_equivalence = 'next_activity'
    model_choice = "CNN"
    root_path = os.getcwd()
    best_res_name = dataset+'_best_results.csv'
    best_res_path = os.path.join(root_path, 'results', dataset, best_res_name)
    best_res = pd.read_csv(best_res_path)
    sel_model = best_res[best_res['Model']==model_choice] 
    sel_task = sel_model[sel_model['Tasks']==task_choice]              
    mtl = get_mtl(dataset)
    stl_df = sel_model[sel_model['Tasks']==single_task] 
    learning_rate = stl_df.iloc[0]['Learning Rate']
    stl_str = model_choice+'__STL_'+single_equivalence+'_task__next_activity_EW_None_'+str(learning_rate)+'_'
    mtl_df = sel_task[sel_task['MTL']==mtl]
    mtl_hpo = "None" if pd.isna(mtl_df.iloc[0]['MTL HPO']) else str(mtl_df.iloc[0]['MTL HPO'])
    learning_rate = mtl_df.iloc[0]['Learning Rate']
    mtl_str = model_choice+'__MTL_'+task_equivalence+'_task__next_activity_'+mtl+'_'+mtl_hpo+'_'+str(learning_rate)+'_'
    stl_name = stl_str+str(seed)+'.csv'
    stl_path = os.path.join(root_path, 'models', dataset, stl_name)
    stl_df = pd.read_csv(stl_path)
    mtl_name = mtl_str+str(seed)+'.csv'
    mtl_path = os.path.join(root_path, 'models', dataset, mtl_name)
    mtl_df = pd.read_csv(mtl_path)
    pdf_path = os.path.join(root_path, dataset+'_class_analysis.pdf')
    compare_model_f1_bar(mtl_df, stl_df, mtl, pdf_path)

if __name__ == '__main__':
    main() 
