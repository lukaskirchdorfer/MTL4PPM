# -*- coding: utf-8 -*-
"""
This script is used to create a plot for a concrete activity class from 
activity classes.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


def main():
    dataset = 'BPIC20_InternationalDeclarations' # 'Production'
    tasks = 'NAP+NTP+RTP'
    model = 'CNN'
    mtl = 'UW'
    act_classes = [4] # activity class of interest    
    class_dict = {4: 'Declaration APPROVED by SUPERVISOR'}
    #class_dict = {26: 'Permit REJECTED by SUPERVISOR'}
    #class_dict = {2: 'Final Inspection Q.C.'}  this one is for Production event log
    
    csv_name = dataset + '_best_results.csv'
    csv_path = os.path.join(os.getcwd(), dataset, csv_name)
    res_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', dataset)
    srch_str, task_str = comb_string(tasks)
    df = get_res(csv_path, srch_str, task_str, res_dir, model, mtl)
    #print(df.head())
    #act_classes = df['next_act'].unique().tolist()
    for target_class in act_classes:
        df['is_target'] = (df['next_act'] == target_class).astype(int)
        target_next_time = df.loc[df['is_target'] == 1, 'next_time']
        other_next_time = df.loc[df['is_target'] == 0, 'next_time']
        target_rem_time = df.loc[df['is_target'] == 1, 'rem_time']
        other_rem_time = df.loc[df['is_target'] == 0, 'rem_time']        
        stat_ntp, p_ntp = ttest_ind(
            target_next_time, other_next_time, equal_var=False)
        stat_rtp, p_rtp = ttest_ind(
            target_rem_time, other_rem_time, equal_var=False)
        print(f"For {target_class} class: t-test for next_time: p = {p_ntp:.4f}, t-test for rem_time: p = {p_rtp:.4f}")
        # Set figure size and layout
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Plot next_time
        sns.boxplot(data=df, x='is_target', y='next_time', hue='is_target',
                    ax=axes[0], palette={0: 'skyblue', 1: 'salmon'}, legend=False)
        axes[0].set_ylim(0, 17)
        axes[0].annotate(f'p-value (t-test): {p_ntp:.3f}', xy=(0.5, 0.95),
                 xycoords='axes fraction', ha='center', va='top',
                 fontsize=14, fontweight='bold')
        #axes[0].set_title(f'next_time vs is_target ({class_dict[target_class]})')
        axes[0].set_xlabel(f'{class_dict[target_class]}',
                           fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Next Time (days)', fontsize=12, fontweight='bold')
        # Plot rem_time
        sns.boxplot(data=df, x='is_target', y='rem_time', hue='is_target',
            ax=axes[1], palette={0: 'skyblue', 1: 'salmon'}, legend=False)
        axes[1].annotate(f'p-value (t-test): {p_rtp:.3f}', xy=(0.5, 0.95),
                 xycoords='axes fraction', ha='center', va='top',
                 fontsize=14, fontweight='bold')
        #axes[1].set_title(f'rem_time vs is_target ({class_dict[target_class]})')
        axes[1].set_xlabel(f'{class_dict[target_class]}', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Remaining Time (days)', fontsize=12, fontweight='bold')
        # Tight layout and save as PDF
        plt.tight_layout()
        save_name = dataset + '_auxiliary_signals_boxplot.pdf'
        plt.savefig(save_name)
        plt.close()
    
    
def get_res(csv_path, srch_str, task_str, res_dir, model, mtl, seed=42):
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
    stl_str = "('next_activity',)"
    stl_df = df_inp[(df_inp['Model'] == model) & (df_inp['Tasks'] == stl_str)].copy()
    stl_df = stl_df.reset_index(drop=True)
    lr_stl = stl_df.loc[0, 'Learning Rate']
    stl_name = model+'__STL_'+'next_activity'+'_task__'+'next_activity'+'_EW_None_'+str(lr_stl)+'_'+str(seed)+'.csv'
    mtl_nap = model+'__MTL_'+task_str+'_task__'+'next_activity'+'_'+mtl+'_'+hpo+'_'+str(lr_mtl)+'_'+str(seed)+'.csv'
    mtl_ntp =  model+'__MTL_'+task_str+'_task__'+'next_time'+'_'+mtl+'_'+hpo+'_'+str(lr_mtl)+'_'+str(seed)+'.csv'
    mtl_rtp =  model+'__MTL_'+task_str+'_task__'+'remaining_time'+'_'+mtl+'_'+hpo+'_'+str(lr_mtl)+'_'+str(seed)+'.csv'
    df_stl = pd.read_csv(os.path.join(res_dir,stl_name))
    df_nap = pd.read_csv(os.path.join(res_dir,mtl_nap))
    df_ntp = pd.read_csv(os.path.join(res_dir,mtl_ntp))
    df_rtp = pd.read_csv(os.path.join(res_dir,mtl_rtp))
    df_combined = pd.DataFrame({
        'case_id': df_stl['case_id'],
        'prefix_length': df_stl['prefix_length'],
        'next_time': df_ntp['ground_truth'],
        'rem_time': df_rtp['ground_truth'],
        'next_act': df_stl['ground_truth'],
        'pred_stl': df_stl['prediction'],
        'pred_mtl': df_nap['prediction']}      
        )
    return df_combined
    
        
    
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
