# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:21:39 2025
@author: kamirel
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import numpy as np

def main():
    dataset_names = ['P2P', 'Production', 'HelpDesk', 'Sepsis', 
                     'BPIC20_DomesticDeclarations',
                     'BPIC20_InternationalDeclarations',
                     'BPI_Challenge_2012C', 'BPI_Challenge_2013_incidents',
                     'BPIC15_1']  
    #task_combinations = ['NAP+NTP+RTP', 'NTP+RTP', 'NAP+NTP', 'NAP+RTP', 'ALL']
    task_combinations = ['ALL']
    #select_combinations = [True, True, True, True, False]
    select_combinations = [False]
    lstm_choice = [True, True, False, False]
    cnn_choice = [True, False, True, False]
    trans_choice = [True, False, False, True]  
    arch_choice = ['ALL', 'LSTM', 'CNN', 'Transformer']
    
    sns.set_theme(context="talk")
    
    mto_order = ['EW', 'DWA' , 'RLW', 'UW', 'UW-SO', 'UW-O', 'GLS', 'GradDrop',
                 'CAGrad', 'PCGrad', 'GradNorm', 'IMTL', 'NashMTL']   
    for combination, select in zip(task_combinations, select_combinations): 
        for lstm, cnn, trans, arch in zip(
                lstm_choice, cnn_choice, trans_choice, arch_choice):
            dataframes, result_nums = [], []
            for name in dataset_names:
                # import csv results
                csv_name = name + '_best_results.csv'
                csv_path = os.path.join(os.getcwd(), name, csv_name)
                df = pd.read_csv(csv_path) 
                df.rename(columns={'MTL': 'MTO'}, inplace=True)
                df['MTO'] = df['MTO'].replace('UW_SO', 'UW-SO')
                df['MTO'] = df['MTO'].replace('UW_O', 'UW-O')
                df['MTO'] = df['MTO'].replace('Nash_MTL', 'NashMTL')
                res_df = add_delta_m(df, combination, select,
                                     lstm=lstm, cnn=cnn, transformer=trans)
                dataframes.append(res_df)
                result_nums.append(len(res_df))
            print(result_nums)
            results = pd.concat(dataframes, ignore_index=True)
            print(len(results))
            #print(results.head())
            if arch == 'ALL':
                plot_model_performance_boxplot(results, combination, select, arch)
            plot_mto_boxplot(results, combination, select, mto_order, arch)            


def plot_model_performance_boxplot(df, combination, select, arch, title=False):
    
    output_pdf_path = combination+'_'+arch+'_architecture_performance.pdf'
    plt.figure(figsize=(5, 4))
    
    ax = sns.boxplot(
        x='Model', y='delta_m', data=df, order=['LSTM', 'CNN', 'Transformer'],
        boxprops=dict(facecolor='none', edgecolor='black'),
        medianprops=dict(color='orange', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='none',
                        markeredgecolor='black', markersize=6, linestyle='none'),
        showcaps=True
    )
    
    for label in ax.get_xticklabels():
        #label.set_fontweight('bold')
        label.set_fontsize(14)
    ax.tick_params(axis='y', labelsize=13)
    # Remove the axes frame (the black box around plot area)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White grid lines
    ax.grid(True, color='white')
    if title:
        # Title and labels with bigger bold font
        ax.set_title(f'Impact of MTL on different models ({combination}, {arch})',
                     fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('')
    ax.set_ylabel(r'$\Delta_m$', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()
    

def plot_mto_boxplot(df, combination, select, mto_order, arch, title=False):
    
    output_pdf_path = combination+'_'+arch+ '_mto_performance.pdf'
    plt.figure(figsize=(9, 6))
    
    ax = sns.boxplot(
        x='MTO', y='delta_m', data=df, order=mto_order,
        boxprops=dict(facecolor='none', edgecolor='black'),
        medianprops=dict(color='orange', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='none',
                        markeredgecolor='black', markersize=6, linestyle='none'),
        showcaps=True
    )
    
    for label in ax.get_xticklabels():
        #label.set_fontweight('bold')
        label.set_rotation(90)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=18) 
    # Remove the axes frame (the black box around plot area)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White grid lines
    ax.grid(True, color='white')
    if title:
        ax.set_title(f'Performance of MTO approaches ({combination}, {arch})',
                     fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('')
    ax.set_ylabel(r'$\Delta_m$', fontsize=30, fontweight='bold')    
    plt.tight_layout()
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()
    
    
def add_delta_m(df_inp, combination, select,
                lstm=True, cnn=True, transformer=True):
    # get single task results for all models
    lstm1, lstm2, lstm3 = get_stl_results(df_inp, model='LSTM')
    cnn1, cnn2, cnn3 = get_stl_results(df_inp, model='CNN')
    trans1, trans2, trans3 = get_stl_results(df_inp, model='Transformer')
    # exclude single task results
    excluded_values = ["('next_activity',)", "('next_time',)", "('remaining_time',)"]    
    df = df_inp[~df_inp['Tasks'].isin(excluded_values)]       
    if select:
        # only focus on an specific task combination
        srch_str = comb_string(combination)
        sel_df = df[df['Tasks'] == srch_str].copy()
    else:
        sel_df = df.copy()
    if lstm:
        lstm_df = sel_df[(sel_df['Model'] == 'LSTM')].copy()
    else:        
        lstm_df = sel_df.iloc[0:0].copy()
    if cnn:
        cnn_df = sel_df[(sel_df['Model'] == 'CNN')].copy()
    else:
        cnn_df = sel_df.iloc[0:0].copy()
    if transformer:
        trans_df = sel_df[(sel_df['Model'] == 'Transformer')].copy()
    else:
        trans_df = sel_df.iloc[0:0].copy()
    #print(f"combination: {combination}")
    #print(f"LSTM rows: {len(lstm_df)}, CNN rows: {len(cnn_df)}, TRANS rows: {len(trans_df)}")
    if not lstm_df.empty:
        lstm_df['delta_m'] = lstm_df.apply(
            lambda row: calc_delta_m(
                lstm1, lstm2, lstm3, 
                row['NEXT_ACTIVITY_mean'],
                row['NEXT_TIME_mean'],
                row['REMAINING_TIME_mean']), axis=1)
    if not cnn_df.empty:
        # TODO: for BPIC2015: CNN, IMTL, NTP + RTP is empty!
        cols = ['NEXT_ACTIVITY_mean', 'NEXT_ACTIVITY_std',
                'NEXT_TIME_mean', 'NEXT_TIME_std',
                'REMAINING_TIME_mean', 'REMAINING_TIME_std']
        cnn_df = cnn_df.dropna(subset=cols, how='all')
        cnn_df['delta_m'] = cnn_df.apply(
            lambda row: calc_delta_m(
                cnn1, cnn2, cnn3, 
                row['NEXT_ACTIVITY_mean'],
                row['NEXT_TIME_mean'],
                row['REMAINING_TIME_mean']
                ), axis=1)
    if not trans_df.empty:
        trans_df['delta_m'] = trans_df.apply(
            lambda row: calc_delta_m(
                trans1, trans2, trans3, 
                row['NEXT_ACTIVITY_mean'],
                row['NEXT_TIME_mean'],
                row['REMAINING_TIME_mean']
                ), axis=1)
    agg_df = pd.concat([lstm_df, cnn_df, trans_df], ignore_index=True)
    cols = ['Model', 'MTO', 'delta_m']
    agg_df = agg_df [cols]	    
    return agg_df

def get_stl_results(df, model='LSTM'):
    sel_df = df[(df['Model'] == model) & (df['MTO'] == 'EW')].copy()
    base_acc_nap = sel_df[sel_df['Tasks'] == "('next_activity',)"]['NEXT_ACTIVITY_mean'].iloc[0]
    base_error_ntp = sel_df[sel_df['Tasks'] == "('next_time',)"]['NEXT_TIME_mean'].iloc[0]
    base_error_rtp = sel_df[sel_df['Tasks'] == "('remaining_time',)"]['REMAINING_TIME_mean'].iloc[0]
    return base_acc_nap, base_error_ntp, base_error_rtp


def calc_delta_m(base_acc_nap: float, base_error_ntp: float, 
                 base_error_rtp: float,
                 acc_nap: float, error_ntp: float, error_rtp: float):
    tasks = []
    if acc_nap is not None and not pd.isna(acc_nap):
        tasks.append(acc_nap)
    if error_ntp is not None and not pd.isna(error_ntp):
        tasks.append(error_ntp)
    if error_rtp is not None and not pd.isna(error_rtp):
        tasks.append(error_rtp)
    term1 = ((-1) * (acc_nap - base_acc_nap) / base_acc_nap 
             if acc_nap is not None and not pd.isna(acc_nap)
             else 0)
    term2 = (1 * (error_ntp - base_error_ntp) / base_error_ntp
             if error_ntp is not None and not pd.isna(error_ntp)
             else 0)
    term3 = (1 * (error_rtp - base_error_rtp) / base_error_rtp 
             if error_rtp is not None and not pd.isna(error_rtp)
             else 0)
    delta_m = (1 / len(tasks) * (term1+term2+term3))
    #return (delta_m * 100).round(2)
    return round(delta_m * 100, 2)
               
def comb_string(combination):
    if combination == 'NAP+NTP+RTP':
        string = "('next_activity', 'next_time', 'remaining_time')"
    elif combination == 'NTP+RTP':
        string = "('next_time', 'remaining_time')"
    elif combination == 'NAP+RTP':
        string = "('next_activity', 'remaining_time')"
    elif combination == 'NAP+NTP':
        string = "('next_activity', 'next_time')"
    else:
        string = 'All_Combinations'
    return string

if __name__ == '__main__':
    main() 