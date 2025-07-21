# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:02:43 2025
@author: kamirel
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

def main():
    dataset = 'Production' # Production BPIC20_InternationalDeclarations
    model = 'CNN'  # CNN LSTM Transformer
    tasks = 'NAP+NTP+RTP'
    focus_task = 'next_activity'
    mtl = 'UW' #['EW', 'DWA' , 'RLW', 'UW', 'UW_SO', 'UW_O', 'GLS', 'GradDrop', 'CAGrad', 'PCGrad', 'GradNorm', 'IMTL', 'Nash_MTL']  
    
    csv_name = dataset + '_best_results.csv'
    csv_path = os.path.join(os.getcwd(), dataset, csv_name)
    res_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', dataset)
    srch_str, task_str = comb_string(tasks)
    #class_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_'+mtl+'_activity_classes.pdf'
    class_f_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_'+mtl+'_activity_classes_with_freq.pdf'
    #length_pdf = dataset+'_'+tasks+'_'+model+'_'+focus_task+'_'+mtl+'_prefix_lengths.pdf'
    
    
    df_merged = get_inf_result(
        csv_path, srch_str, task_str, model, mtl, focus_task, res_dir)
    
    #comp_by_class(df_merged, mtl, class_pdf)
    comp_by_class_freq(df_merged, mtl, class_f_pdf)
    #com_by_length(df_merged, mtl, length_pdf)
    #print(df_merged.head())

    
def get_inf_result(
        csv_path, srch_str, task_str, model, mtl, focus_task, res_dir, seed=42):    
    df_inp = pd.read_csv(csv_path)
    df_task = df_inp[df_inp['Tasks'] == srch_str]
    df_model = df_task[df_task['Model'] == model]
    res_df = df_model[df_model['MTL'] == mtl].copy()
    res_df = res_df.reset_index(drop=True)
    lr_mtl = res_df.loc[0, 'Learning Rate']
    if pd.isna(res_df.loc[0, 'MTL HPO']):
        hpo = 'None'
    else:
        hpo = str(res_df.loc[0, 'MTL HPO'])
    stl_str = f"('{focus_task}',)"
    stl_df = df_inp[(df_inp['Model'] == model) & (df_inp['Tasks'] == stl_str)].copy()
    stl_df = stl_df.reset_index(drop=True)
    lr_stl = stl_df.loc[0, 'Learning Rate']
    stl_name = model+'__STL_'+focus_task+'_task__'+focus_task+'_EW_None_'+str(lr_stl)+'_'+str(seed)+'.csv'
    mtl_name = model+'__MTL_'+task_str+'_task__'+focus_task+'_'+mtl+'_'+hpo+'_'+str(lr_mtl)+'_'+str(seed)+'.csv'
    df_stl = pd.read_csv(os.path.join(res_dir,stl_name))
    df_mtl = pd.read_csv(os.path.join(res_dir,mtl_name))
    df_merged = df_stl.merge(
        df_mtl,
        on=['case_id', 'prefix_length', 'ground_truth'],
        suffixes=('_stl', '_mtl'))
    
    """
    selected_prefixes = df_merged[
        (df_merged['ground_truth'] == 2) & 
        (df_merged['prediction_stl'] != 2) & 
        (df_merged['prediction_mtl'] == 2)][['case_id', 'prefix_length']]
    sel_pr = list(selected_prefixes.itertuples(index=False, name=None))
    print(len(sel_pr))
    gt_list = [
        df_merged[
            (df_merged['case_id'] == case_id) & 
            (df_merged['prefix_length'] == pl - 1)]['ground_truth'].values[0]
        for case_id, pl in sel_pr
        if not df_merged[(df_merged['case_id'] == case_id) & 
                         (df_merged['prefix_length'] == pl - 1)].empty]
    print(gt_list)
    print(len(gt_list))
    """
    
    return df_merged
    
    
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


def comp_by_class(df, mtl, class_pdf):
    results = []
    classes = sorted(df['ground_truth'].unique())
    for act in classes:
        subset = df[df['ground_truth'] == act]
        acc_stl = accuracy_score(subset['ground_truth'], subset['prediction_stl'])
        acc_mtl = accuracy_score(subset['ground_truth'], subset['prediction_mtl'])
        results.append({'class': act, 'model': 'STL', 'accuracy': acc_stl})
        results.append({'class': act, 'model': mtl, 'accuracy': acc_mtl})
    results_df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x='class', y='accuracy', hue='model')
    plt.title(f'Per-Class NAP Accuracy: STL vs {mtl}')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Acitivty Class')
    plt.legend(title='Model')
    plt.tight_layout()    
    plt.savefig(class_pdf)
    plt.close()

def comp_by_class_freq(df, mtl, class_f_pdf):
    classes = sorted(df['ground_truth'].unique())
    results = []
    class_counts = df['ground_truth'].value_counts().to_dict()
    for act in classes:
        subset = df[df['ground_truth'] == act]
        acc_stl = accuracy_score(subset['ground_truth'], subset['prediction_stl'])
        acc_mtl = accuracy_score(subset['ground_truth'], subset['prediction_mtl'])
        results.append({'class': act, 'model': 'STL', 'accuracy': acc_stl, 'count': class_counts[act]})
        results.append({'class': act, 'model': mtl, 'accuracy': acc_mtl, 'count': class_counts[act]})
    results_df = pd.DataFrame(results)
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_df, x='class', y='accuracy', hue='model', ax=ax1)
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Acitivty Class')
    ax1.set_ylim(0, 1)
    ax1.set_title(f'Per-Class NAP Accuracy: STL vs {mtl}')

    # Secondary axis for class frequency
    ax2 = ax1.twinx()
    class_order = sorted(df['ground_truth'].unique())
    class_freq = df['ground_truth'].value_counts(normalize=True).reindex(class_order)
    ax2.plot(range(len(class_order)), class_freq.values, color='gray',
             marker='o', linestyle='--', label='Class Frequency')
    ax2.set_xticks(range(len(class_order)))
    ax2.set_xticklabels(class_order)
    ax2.set_ylabel('Relative Frequency')
    ax2.legend(loc='upper right')

    ax1.legend(title='Model', loc='upper left')
    plt.tight_layout()
    plt.savefig(class_f_pdf)
    plt.close()


def com_by_length(df, mtl, length_pdf):
    prefix_lengths = sorted(df['prefix_length'].unique())
    results = []
    for length in prefix_lengths:
        subset = df[df['prefix_length'] == length]
        acc_stl = accuracy_score(subset['ground_truth'], subset['prediction_stl'])
        acc_mtl = accuracy_score(subset['ground_truth'], subset['prediction_mtl'])
        results.append({'prefix_length': length, 'model': 'STL', 'accuracy': acc_stl})
        results.append({'prefix_length': length, 'model': mtl, 'accuracy': acc_mtl})
    results_df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=results_df, x='prefix_length', y='accuracy', hue='model')
    plt.title('NAP Accuracy vs. Prefix Length')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Prefix Length')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(length_pdf)
    plt.close() 

    

if __name__ == '__main__':
    main()
    