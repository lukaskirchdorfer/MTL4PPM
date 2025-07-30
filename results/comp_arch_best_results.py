# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 08:52:29 2025
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    datasets = ['P2P', 'Production', 'HelpDesk', 'Sepsis',  'BPIC15_1',
                'BPIC20_DomesticDeclarations',
                'BPIC20_InternationalDeclarations', 'BPI_Challenge_2012C',
                'BPI_Challenge_2013_incidents'] 
    models = ['LSTM', 'CNN', 'Transformer']
    combs = ['NAP+NTP+RTP', 'NTP+RTP', 'NAP+RTP', 'NAP+NTP']
    combs = ['NAP+NTP+RTP']
    
    for name in datasets:
        out_pdf = name+'_best_results.pdf'
        # import csv results
        csv_name = name + '_best_results.csv'
        csv_path = os.path.join(os.getcwd(), name, csv_name)
        df = pd.read_csv(csv_path) 
        # get stl results
        lstm_nap_stl, lstm_ntp_stl, lstm_rtp_stl = get_stl_results(
            df, model='LSTM')
        cnn_nap_stl, cnn_ntp_stl, cnn_rtp_stl = get_stl_results(
            df, model='CNN')
        trans_nap_stl, trans_ntp_stl, trans_rtp_stl = get_stl_results(
            df, model='Transformer')
        # exclude single task results
        excluded_values = ["('next_activity',)", "('next_time',)", "('remaining_time',)"]    
        df_mtl = df[~df['Tasks'].isin(excluded_values)].copy()
        with PdfPages(out_pdf) as pdf:
            fig, axes = plt.subplots(len(combs), 3,
                                     figsize=(10, 3*len(combs)))
            for k, comb in enumerate(combs):
                srch_str = comb_string(comb)
                sel_df = df_mtl[df_mtl['Tasks'] == srch_str].copy()
                lstm_df = sel_df[sel_df['Model'] == 'LSTM'].copy()
                lstm_nap_mtl = lstm_df['NEXT_ACTIVITY_mean'].max()
                lstm_ntp_mtl = lstm_df['NEXT_TIME_mean'].min()
                lstm_rtp_mtl = lstm_df['REMAINING_TIME_mean'].min()
                cnn_df = sel_df[sel_df['Model'] == 'CNN'].copy()
                cnn_nap_mtl = cnn_df['NEXT_ACTIVITY_mean'].max()
                cnn_ntp_mtl = cnn_df['NEXT_TIME_mean'].min()
                cnn_rtp_mtl = cnn_df['REMAINING_TIME_mean'].min()
                trans_df = sel_df[sel_df['Model'] == 'Transformer'].copy()
                trans_nap_mtl = trans_df['NEXT_ACTIVITY_mean'].max()
                trans_ntp_mtl = trans_df['NEXT_TIME_mean'].min()
                trans_rtp_mtl = trans_df['REMAINING_TIME_mean'].min()
                
                # NAP sub-plot
                values = [lstm_nap_stl, cnn_nap_stl, trans_nap_stl, None,
                          lstm_nap_mtl, cnn_nap_mtl, trans_nap_mtl]
                colors = ['skyblue', 'salmon', 'lightgreen', 'white',
                          'skyblue', 'salmon', 'lightgreen']
                labels = ['LSTM', 'CNN', 'Transformer', '',
                          'LSTM', 'CNN', 'Transformer']
                x = np.arange(len(values)) # Bar positions
                if len(combs) > 1:
                    ax = axes[k, 0]
                else:
                    ax = axes[0]
                # Plot bars
                for i, val in enumerate(values):
                    if val is not None:
                        ax.bar(x[i], val, color=colors[i], label=labels[i] if i < 3 else "")
                # Group labels
                ax.set_xticks([1, 5])
                ax.set_xticklabels(['STL', 'Best MTL'])
                if k == 0:
                    handles = [plt.Rectangle((0,0),1,1,color=c) for c in ['skyblue', 'salmon', 'lightgreen']]
                    ax.legend(handles, ['LSTM', 'CNN', 'Transformer'])
                ax.set_xlabel(f'NAP- combination:{comb}')   
                ax.set_ylabel('Accuracy')
                ax.grid(axis='y') 
                
                # NTP subplot
                values = [lstm_ntp_stl, cnn_ntp_stl, trans_ntp_stl, None,
                          lstm_ntp_mtl, cnn_ntp_mtl, trans_ntp_mtl]
                colors = ['skyblue', 'salmon', 'lightgreen', 'white',
                          'skyblue', 'salmon', 'lightgreen']
                labels = ['LSTM', 'CNN', 'Transformer', '',
                          'LSTM', 'CNN', 'Transformer']
                x = np.arange(len(values)) # Bar positions
                if len(combs) > 1:
                    ax = axes[k, 1]
                else:
                    ax = axes[1]
                # Plot bars
                for i, val in enumerate(values):
                    if val is not None:
                        ax.bar(x[i], val, color=colors[i], label=labels[i] if i < 3 else "")
                # Group labels
                ax.set_xticks([1, 5])
                ax.set_xticklabels(['STL', 'Best MTL'])
                ax.set_xlabel(f'NTP- combination:{comb}')   
                ax.set_ylabel('MAE')
                ax.grid(axis='y')  
                
                # RTP subplot
                values = [lstm_rtp_stl, cnn_rtp_stl, trans_rtp_stl, None,
                          lstm_rtp_mtl, cnn_rtp_mtl, trans_rtp_mtl]
                colors = ['skyblue', 'salmon', 'lightgreen', 'white',
                          'skyblue', 'salmon', 'lightgreen']
                labels = ['LSTM', 'CNN', 'Transformer', '',
                          'LSTM', 'CNN', 'Transformer']
                x = np.arange(len(values)) # Bar positions
                if len(combs) > 1:
                    ax = axes[k, 2]
                else:
                    ax = axes[2]
                # Plot bars
                for i, val in enumerate(values):
                    if val is not None:
                        ax.bar(x[i], val, color=colors[i], label=labels[i] if i < 3 else "")
                # Group labels
                ax.set_xticks([1, 5])
                ax.set_xticklabels(['STL', 'Best MTL'])
                ax.set_xlabel(f'RTP- combination:{comb}')   
                ax.set_ylabel('MAE')
                ax.grid(axis='y')  
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()  

    
def get_stl_results(df, model='LSTM'):
    sel_df = df[(df['Model'] == model) & (df['MTL'] == 'EW')].copy()
    base_acc_nap = sel_df[sel_df['Tasks'] == "('next_activity',)"]['NEXT_ACTIVITY_mean'].iloc[0]
    base_error_ntp = sel_df[sel_df['Tasks'] == "('next_time',)"]['NEXT_TIME_mean'].iloc[0]
    base_error_rtp = sel_df[sel_df['Tasks'] == "('remaining_time',)"]['REMAINING_TIME_mean'].iloc[0]
    return base_acc_nap, base_error_ntp, base_error_rtp

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