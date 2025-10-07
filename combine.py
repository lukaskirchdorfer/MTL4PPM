# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 17:14:02 2025

@author: kamirel
"""
import os
import pandas as pd

def main():
    root_path = os.getcwd()
    file1 = os.path.join(root_path, 'negative_transfer_1.csv')
    file2 = os.path.join(root_path, 'negative_transfer_2.csv')
    res1 = pd.read_csv(file1)
    res2 = pd.read_csv(file2)
    df = pd.concat([res1, res2], ignore_index=True)
    csv_path = os.path.join(root_path, 'negative_transfer.csv')
    df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    main() 
    