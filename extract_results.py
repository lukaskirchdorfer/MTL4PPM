# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:20:58 2025
@author: kamirel
"""

import os
import pandas as pd
import re
import argparse

def parse_log_files(log_dir):
    results = []
    for file in os.listdir(log_dir):
        if file.endswith(".log"):
            full_path = os.path.join(log_dir, file)
            # Parse filename structure
            parts = file[:-4].split('_')  # remove .log and split
            if len(parts) < 3:
                continue  # skip malformed filenames
            model = parts[0]
            mtl = parts[-1]
            tasks = [parts[1:-1][i] + "_" + parts[1:-1][i+1] 
                     for i in range(0, len(parts[1:-1]) - 1, 2)]
            tasks = [x for x in tasks 
                     if x in ['next_activity', 'next_time', 'remaining_time']]
            # Default values
            metrics = {
                'Model': model,
                'MTL': mtl,
                'NEXT_ACTIVITY': float('nan'),
                'NEXT_TIME': float('nan'),
                'REMAINING_TIME': float('nan')
            }
            try:
                with open(full_path, 'r') as f:
                    lines = f.readlines()[-3:]  # last three lines                    
                if (len(lines) < 3 or 
                    "Now: Inference on Test set." not in lines[-3]):
                    results.append(metrics)
                    continue
                # The third-to-last should be "Inference" log, skip
                # The second-to-last has the column labels (optional, ignored here)
                # The last line has the actual values
                last_line = lines[-1]
                # Extract all floating point numbers
                #values = list(map(float, re.findall(r"\d+\.\d+", last_line)))
                values_line_clean = last_line.split(" - ")[-1]
                values = [float(v.strip()) for v in values_line_clean.split('|') if v.strip()]
                # Map values to tasks if available
                #for task, value in zip(tasks, values):
                    #col = task.upper()
                    #if col in metrics:
                        #metrics[col] = value
                print(tasks, values)
                for task_name, val in zip(tasks, values):
                    key = task_name.strip().upper()
                    metrics[key] = val
            except Exception as e:
                print(f"Error reading {file}: {e}")
            results.append(metrics)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='collect results')    
    parser.add_argument('--dataset', type=str, required=True, 
                        help='name of the dataset')
    args = parser.parse_args()
    root_path = os.getcwd()
    log_dir = os.path.join(root_path, 'models', args.dataset)   
    df = parse_log_files(log_dir)
    csv_path = os.path.join(log_dir, args.dataset+'_overall_results.csv')
    df.to_csv(csv_path, index=False)
    
if __name__ == '__main__':
    main() 