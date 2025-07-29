# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:47:11 2025
@author: kamirel
This script is used to create table 2 of the paper: i.e., the wilcoxon test for
three different prediction tasks (NAP, NTP, RTP) separately for three different
architectures.  For Transformer, we need to exclude BPI_Challenge_2012C, since
currently the results for this log are not available.
"""
import os
import pandas as pd
from scipy.stats import wilcoxon

def main():
    datasets = ['P2P', 'Production', 'HelpDesk', 'Sepsis',  'BPIC15_1',
                'BPIC20_DomesticDeclarations',
                'BPIC20_InternationalDeclarations',
                'BPI_Challenge_2013_incidents', 'BPI_Challenge_2012C'] 
    # 'BPIC15_1', 'BPI_Challenge_2013_incidents', 'BPI_Challenge_2012C'
    model = 'LSTM' #'CNN' 'LSTM' 'Transformer'
    task_comb = 'NAP+NTP+RTP' #'NTP+RTP' 'NAP+NTP' 'NAP+RTP' 'NAP+NTP+RTP'
    task_comb = 'ALL'
    stl_nap_lst, stl_ntp_lst, stl_rtp_lst = [], [], []
    mtl_nap_lst, mtl_ntp_lst, mtl_rtp_lst = [], [], []
    for name in datasets:
        # import csv results
        csv_name = name + '_best_results.csv'
        csv_path = os.path.join(os.getcwd(), name, csv_name)
        df = pd.read_csv(csv_path) 
        result = get_stl_mtl(df, model, task_comb)
        stl_nap_lst.append(result[0])
        stl_ntp_lst.append(result[1])
        stl_rtp_lst.append(result[2])
        mtl_nap_lst.append(result[3])
        mtl_ntp_lst.append(result[4])
        mtl_rtp_lst.append(result[5])
    # wilcoxon signed-ranked test
    # Next Activity Prediction
    stl_scores = stl_nap_lst
    mtl_scores = mtl_nap_lst
    stat, p = wilcoxon(stl_scores, mtl_scores)
    print('Results for NAP', stat, p)
    # Next Time Prediction
    stl_scores = [-x for x in stl_ntp_lst]
    mtl_scores = [-x for x in mtl_ntp_lst]
    stat, p = wilcoxon(stl_scores, mtl_scores)
    print('Results for NTP', stat, p)
    # Remaining Time Prediction
    stl_scores = [-x for x in stl_rtp_lst]
    mtl_scores = [-x for x in mtl_rtp_lst]
    stat, p = wilcoxon(stl_scores, mtl_scores)
    print('Results for RTP', stat, p)
    
       
        
def get_stl_mtl(df_inp, model, task_comb):
    
    srch_str = comb_string(task_comb)
    nap_str = "('next_activity',)"
    ntp_str = "('next_time',)"
    rtp_str = "('remaining_time',)"
    
    df_model = df_inp[df_inp['Model'] == model].copy()
    if srch_str == 'All_Combinations':        
        excluded_values = ["('next_activity',)", "('next_time',)", "('remaining_time',)"]
        df_comb = df_model[~df_model['Tasks'].isin(excluded_values)]
    else:    
        df_comb = df_model[df_model['Tasks'] == srch_str].copy()
    df_nap = df_model[df_model['Tasks'] == nap_str].copy()
    df_ntp = df_model[df_model['Tasks'] == ntp_str].copy()
    df_rtp = df_model[df_model['Tasks'] == rtp_str].copy()
    print(len(df_comb), len(df_nap), len(df_ntp), len(df_rtp))
    
    nap_stl = df_nap['NEXT_ACTIVITY_mean'].mean()
    nap_mtl = df_comb['NEXT_ACTIVITY_mean'].mean()
    ntp_stl = df_ntp['NEXT_TIME_mean'].mean()
    ntp_mtl = df_comb['NEXT_TIME_mean'].mean()
    rtp_stl = df_rtp['REMAINING_TIME_mean'].mean()
    rtp_mtl = df_comb['REMAINING_TIME_mean'].mean()
    
    result = (nap_stl, ntp_stl, rtp_stl, nap_mtl, ntp_mtl, rtp_mtl)
    #print(nap_stl)
    #print(nap_mtl)
    print((rtp_stl-rtp_mtl)/rtp_mtl)
    return result

        
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