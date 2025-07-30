# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:47:01 2025
"""
import os
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
import numpy as np

def main():
    datasets = ['P2P', 'Production', 'HelpDesk', 'Sepsis', 'BPIC15_1', 
                'BPIC20_DomesticDeclarations',
                'BPIC20_InternationalDeclarations'] 
    # 'BPI_Challenge_2012C', 'BPI_Challenge_2013_incidents', 
    mtl_order = ['EW', 'DWA' , 'RLW', 'UW', 'UW_SO', 'UW_O', 'GLS', 'GradDrop',
                 'CAGrad', 'PCGrad', 'GradNorm', 'IMTL', 'Nash_MTL']  
    lstm_lst, cnn_lst, trans_lst = [], [], []
    mtl_lst = [[] for _ in range(13)]
    for name in datasets:
        print(name)
        # import csv results
        csv_name = name + '_best_results.csv'
        csv_path = os.path.join(os.getcwd(), name, csv_name)
        df = pd.read_csv(csv_path) 
        # get STL results for different architectures
        lstm1, lstm2, lstm3 = get_stl_results(df, model='LSTM')
        cnn1, cnn2, cnn3 = get_stl_results(df, model='CNN')
        trans1, trans2, trans3 = get_stl_results(df, model='Transformer')
        # get MTL results and divide it based on architecture
        excluded_values = ["('next_activity',)", "('next_time',)", "('remaining_time',)"]
        df_mtl = df[~df['Tasks'].isin(excluded_values)]
        lstm_df = df_mtl[(df_mtl['Model'] == 'LSTM')].copy()
        cnn_df = df_mtl[(df_mtl['Model'] == 'CNN')].copy()
        trans_df = df_mtl[(df_mtl['Model'] == 'Transformer')].copy()
        # add delta M values
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
        # get delta M average for each architecture
        lstm_lst.append(lstm_df['delta_m'].mean())
        cnn_lst.append(cnn_df['delta_m'].mean())
        trans_lst.append(trans_df['delta_m'].mean())
        agg_df = pd.concat([lstm_df, cnn_df, trans_df], ignore_index=True)
        for i , approach in enumerate(mtl_order):
            subset = agg_df[(agg_df['MTL'] == approach)].copy()
            mtl_lst[i].append(subset['delta_m'].mean())

    lstm_scores = [-x for x in lstm_lst]
    cnn_scores = [-x for x in cnn_lst]
    trans_scores = [-x for x in trans_lst]
    stat, p = wilcoxon(lstm_scores, cnn_scores)
    print('LSTM vs. CNN:', stat, p)
    stat, p = wilcoxon(lstm_scores, trans_scores)
    print('LSTM vs. Transformer:', stat, p)
    stat, p = wilcoxon(cnn_scores, trans_scores)
    print('CNN vs. Transformer:', stat, p)
    mtl_scores = [[-x for x in sublist] for sublist in mtl_lst]
    print(mtl_scores)
    evaluate_mtl_methods(mtl_order, mtl_scores)
    
    

def get_stl_results(df, model='LSTM'):
    sel_df = df[(df['Model'] == model) & (df['MTL'] == 'EW')].copy()
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



def evaluate_mtl_methods(mtl_order, mtl_scores):
    # Transpose scores to shape: (n_datasets, n_methods)
    scores_by_dataset = np.array(mtl_scores).T    
    # Step 1: Friedman test across all methods
    stat, p = friedmanchisquare(*scores_by_dataset.T)
    print("Friedman Test (Overall Difference):")
    print(f"chi-2 = {stat:.3f}, p = {p:.4f}")
    if p < 0.05:
        print("Significant differences found among methods.")
    else:
        print("No significant differences found among methods.")
    # Step 2: Pairwise Wilcoxon tests against 'EW'
    print("Wilcoxon Signed-Rank Test (Each vs 'EW'):")
    ref_idx = mtl_order.index('EW')
    ew_scores = scores_by_dataset[:, ref_idx]
    for i, method in enumerate(mtl_order):
        if method == 'EW':
            continue
        stat, p = wilcoxon(scores_by_dataset[:, i], ew_scores)
        better = "Better than" if p < 0.05 and np.mean(scores_by_dataset[:, i]) > np.mean(ew_scores) else "Not significantly better than"
        print(f"{method:10s}: W = {stat:.2f}, p = {p:.4f} → {better} 'EW'")


if __name__ == '__main__':
    main() 