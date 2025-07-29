# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:47:14 2025
@author: kamirel
This script is used to extract training times for different MTO strategies.
It can be used in two modes:
    1) If mode: collect is False then it extracts the training times for
    different MTO strategies for one dataset. 
    2) If mode: collect is True, then it uses already extracted training times
    for different datasets to make an overal dataframe for training time in all
    datasets.
"""
import os
import argparse
import pandas as pd
from datetime import datetime

def main():
    tasks = 'NAP+NTP+RTP'
    model = 'CNN'
    mtls = ['EW', 'DWA' , 'RLW', 'UW', 'UW_SO', 'UW_O', 'GLS', 'GradDrop',
            'CAGrad', 'PCGrad', 'GradNorm', 'IMTL', 'Nash_MTL']
    parser = argparse.ArgumentParser(description='Training Time Extraction')
    parser.add_argument('--dataset', type=str, default='Production',
                        help='name of the dataset')
    parser.add_argument("--collect", action="store_true", default=False, 
                        help="collect info for multiple datasets")
    args = parser.parse_args() 
    if args.collect:
        datasets = ['BPIC20_DomesticDeclarations',
                    'BPIC20_InternationalDeclarations','BPIC15_1', 
                    'BPI_Challenge_2013_incidents', 'BPI_Challenge_2012C',
                    'HelpDesk', 'Sepsis', 'P2P', 'Production']
        labels = ['BPIC20DD', 'BPIC20ID', 'BPIC15-1', 'BPIC13I', 'BPIC12C',
                  'Helpdesk', 'Sepsis', 'P2P', 'Production']
        df_lst = []
        for dataset in datasets:
            csv_name = dataset + '_train_time.csv'
            csv_path = os.path.join(os.getcwd(), dataset, csv_name)
            df_time = pd.read_csv(csv_path)
            df_lst.append(df_time)
        agg_df = combine_training_times(df_lst, labels, mtls)
        agg_path = os.path.join(os.getcwd(), 'overall_train_time.csv')
        agg_df.to_csv(agg_path, index=False)      
    else:
        csv_name = args.dataset + '_best_results.csv'
        csv_path = os.path.join(os.getcwd(), args.dataset, csv_name)
        res_dir = os.path.join(
            os.path.dirname(os.getcwd()), 'models', args.dataset)
        srch_str, task_str = comb_string(tasks)
        stl_log_lst , mtl_log_lst, stl_epochs, mtl_epochs = get_log_files(
            csv_path, mtls, model, srch_str, task_str, res_dir)
        stl_dur = 0
        mtl_durs = []
        for log, epoch in zip(stl_log_lst, stl_epochs):
            dur = compute_duration(log)
            dur_epoch = dur/epoch
            stl_dur += dur_epoch
        for log, epoch in zip(mtl_log_lst, mtl_epochs):
            dur = compute_duration(log)
            dur_epoch = dur/epoch
            mtl_durs.append(dur_epoch)
        mtls.insert(0, 'STL')
        mtl_durs.insert(0, stl_dur)
        dur_df = pd.DataFrame({'MTL': mtls, 'Duration': mtl_durs})
        #print(dur_df.head(14))
        dur_name = args.dataset+'_train_time.csv'
        dur_path = os.path.join(os.getcwd(), args.dataset, dur_name)
        dur_df.to_csv(dur_path, index=False)
    

def combine_training_times(df_lst, labels, mtls):
    combined_rows = []
    for df, label in zip(df_lst, labels):
        # Add a dummy index for pivoting
        df_new = df.copy()
        df_new["dummy"] = 0
        pivot_df = df_new.pivot(index="dummy", columns="MTL", values="Duration")
        pivot_df["Dataset"] = label
        combined_rows.append(pivot_df)
    final_df = pd.concat(combined_rows, ignore_index=True)
    mtls.insert(0, 'STL')
    cols = ["Dataset"] + mtls
    #cols = ["Dataset"] + [col for col in final_df.columns if col != "Dataset"]
    final_df = final_df[cols]
    final_df.update(final_df.select_dtypes(include='number').round(0).astype(int))
    return final_df



def compute_duration(log_path):
    with open(log_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Extract first and last non-empty lines
    first_line = lines[0]
    last_line = lines[-1]
    # Parse timestamps (first 23 characters)
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    t_start = datetime.strptime(first_line[:23], fmt)
    t_end = datetime.strptime(last_line[:23], fmt)
    # Compute time difference in seconds
    duration = (t_end - t_start).total_seconds()
    return duration
    
    
def get_log_files(csv_path, mtls, model, srch_str, task_str, res_dir, seed=42):
    df_inp = pd.read_csv(csv_path)
    # get log files for STL
    stl_log_lst, stl_epochs = [], []
    nap_str = "('next_activity',)"
    stl_nap = df_inp[(df_inp['Model'] == model) & (df_inp['Tasks'] == nap_str)].copy()
    stl_nap = stl_nap.reset_index(drop=True)
    lr_nap = stl_nap.loc[0, 'Learning Rate']
    nap_epoch = stl_nap.loc[0, 'Best Epoch_mean']
    stl_epochs.append(nap_epoch)
    nap_name = model+'_next_activity_EW_None_'+str(lr_nap)+'_'+str(seed)+'.log'
    stl_log_lst.append(os.path.join(res_dir,nap_name))
    ntp_str = "('next_time',)"
    stl_ntp = df_inp[(df_inp['Model'] == model) & (df_inp['Tasks'] == ntp_str)].copy()
    stl_ntp = stl_ntp.reset_index(drop=True)
    lr_ntp = stl_ntp.loc[0, 'Learning Rate']
    ntp_epoch = stl_ntp.loc[0, 'Best Epoch_mean']
    stl_epochs.append(ntp_epoch)
    ntp_name = model+'_next_time_EW_None_'+str(lr_ntp)+'_'+str(seed)+'.log'
    stl_log_lst.append(os.path.join(res_dir,ntp_name))
    rtp_str = "('remaining_time',)"
    stl_rtp = df_inp[(df_inp['Model'] == model) & (df_inp['Tasks'] == rtp_str)].copy()
    stl_rtp = stl_rtp.reset_index(drop=True)
    lr_rtp = stl_rtp.loc[0, 'Learning Rate']
    rtp_epoch = stl_rtp.loc[0, 'Best Epoch_mean']
    stl_epochs.append(rtp_epoch)
    rtp_name = model+'_remaining_time_EW_None_'+str(lr_rtp)+'_'+str(seed)+'.log'
    stl_log_lst.append(os.path.join(res_dir,rtp_name))
    # get log files for MTL
    mtl_log_lst, mtl_epochs = [], []
    df_task = df_inp[df_inp['Tasks'] == srch_str]
    df_model = df_task[df_task['Model'] == model]
    for mtl in mtls:
        res_df = df_model[df_model['MTL'] == mtl].copy()
        res_df = res_df.reset_index(drop=True)
        lr_mtl = res_df.loc[0, 'Learning Rate']
        if pd.isna(res_df.loc[0, 'MTL HPO']):
            hpo = 'None'
        else:
            hpo = str(res_df.loc[0, 'MTL HPO'])
        mtl_epoch = res_df.loc[0, 'Best Epoch_mean']
        mtl_epochs.append(mtl_epoch)
        mtl_name = model+'_'+task_str+'_'+mtl+'_'+hpo+'_'+str(lr_mtl)+'_'+str(seed)+'.log'
        mtl_log_lst.append(os.path.join(res_dir,mtl_name))
    return stl_log_lst , mtl_log_lst, stl_epochs, mtl_epochs 
    
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