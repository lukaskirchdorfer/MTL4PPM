# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 11:28:39 2025
@author: Keyvan Amiri Elyasi
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    datasets = ['Production', 'BPIC20_InternationalDeclarations', 'BPIC15_1']
    labels = ['Production', 'BPIC20_ID', 'BPIC15_1']
    keys = ['next_activity_vs_next_time', 'next_activity_vs_remaining_time',
            'next_time_vs_remaining_time']
    
    root_path = os.getcwd()
    csv_path = os.path.join(root_path, 'negative_transfer.csv')
    df = pd.read_csv(csv_path)
    for key in keys:
        key_df = df[df['task_pair']==key]
        degree_lst, magnitude_lst = [], []
        for dataset in datasets:
            data_df = key_df[key_df['dataset']==dataset]             
            mean_degree = data_df['Average_degree'].mean()
            mean_magnitude = data_df['average_magnitude'].mean()
            degree_lst.append(mean_degree)
            magnitude_lst.append(mean_magnitude)
        plt.bar(labels, degree_lst)
        plt.xticks(rotation=45)
        plt.axhline(90, color='red', linestyle='--')
        plot_path = os.path.join(root_path, 'gradient_degree_' + key + '.pdf')
        plt.savefig(plot_path, format="pdf", dpi=600, bbox_inches='tight')  
        plt.close()          
            

if __name__ == '__main__':
    main() 