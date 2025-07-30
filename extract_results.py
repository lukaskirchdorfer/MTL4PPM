# -*- coding: utf-8 -*-

import os
import pandas as pd
import argparse

def parse_log_files(log_dir):
    results = []
    best_results = []
    for file in os.listdir(log_dir):
        if file.endswith(".log"):
            full_path = os.path.join(log_dir, file)
            # Parse filename structure
            parts = file[:-4].split('_')  # remove .log and split
            if len(parts) < 3:
                continue  # skip malformed filenames
            model = parts[0]
            mtl = parts[-4]
            if mtl == 'MTL':
                mtl = 'Nash_MTL'
            elif mtl == 'O':
                mtl = 'UW_O'
            elif mtl == 'SO':
                mtl = 'UW_SO'
            tasks = [parts[1:-1][i] + "_" + parts[1:-1][i+1] 
                     for i in range(0, len(parts[1:-1]) - 1, 2)]
            tasks = [x for x in tasks 
                     if x in ['next_activity', 'next_time', 'remaining_time']]
            learning_rate = parts[-2]
            seed = parts[-1]
            mtl_hpo = parts[-3]
            # Default values
            metrics = {
                'Model': model,
                'MTL': mtl,
                'Tasks': tasks,
                'Learning Rate': learning_rate,
                'MTL HPO': mtl_hpo,
                'Seed': seed,
                'Best Epoch': None,
                'Best Validation Loss': float('nan'),
                'NEXT_ACTIVITY': float('nan'),
                'NEXT_TIME': float('nan'),
                'REMAINING_TIME': float('nan')
            }
            try:
                with open(full_path, 'r') as f:
                    lines = f.readlines()[-5:]  # last four lines                    
                if (len(lines) < 5 or 
                    "Now: Inference on Test set." not in lines[-4]):
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
                for task_name, val in zip(tasks, values):
                    key = task_name.strip().upper()
                    metrics[key] = val

                # Extract best validation loss and epoch
                best_val_loss_line = lines[-5]
                best_val_loss = float(best_val_loss_line.split(" ")[-4])
                best_epoch = int(best_val_loss_line.split(" ")[-1])
                metrics['Best Validation Loss'] = best_val_loss
                metrics['Best Epoch'] = best_epoch
            except Exception as e:
                print(f"Error reading {file}: {e}")
            results.append(metrics)
    df = pd.DataFrame(results) 
    # Convert 'Tasks' column from list to tuple for hashability
    if 'Tasks' in df.columns:
        df['Tasks'] = df['Tasks'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    best_results_df = get_best_result_per_model(df)
    # best_results_df = pd.DataFrame()
    # sorting the dataframe
    metric_cols = ['NEXT_ACTIVITY', 'NEXT_TIME', 'REMAINING_TIME']
    # Create a binary signature string
    # like '110' meaning: [NEXT_ACTIVITY, NEXT_TIME, REMAINING_TIME] are [notna, notna, na]
    df['metric_signature'] = df[metric_cols].notna().astype(int).astype(str).agg(''.join, axis=1)
    # Sort by Model and then by the metric signature
    df_sorted = df.sort_values(by=['Model', 'metric_signature'], ascending=[True, False])    
    df_sorted = df_sorted.drop(columns='metric_signature')
    return df_sorted, best_results_df

def get_best_result_per_model(df):
    """
    The extracted dataframe containing all results is analyzed to get the best result per model/task/MTL method.
    """
    best_results = []
    for model in df['Model'].unique():
        for task in df['Tasks'].unique():
            for mtl in df['MTL'].unique():
                df_model_task_mtl = df[(df['Model'] == model) & (df['Tasks'] == task) & (df['MTL'] == mtl)]
                
                # Check if there are any results for this combination
                if df_model_task_mtl.empty:
                    continue  # Skip this combination if no data exists
                
                # groupby learning rate
                df_model_task_mtl = df_model_task_mtl.groupby(['Learning Rate', 'MTL HPO'])
                # for this learning rate, compute the mean and std over the seeds for the metrics NEXT_ACTIVITY, NEXT_TIME, REMAINING_TIME, Best Epoch
                df_model_task_mtl = df_model_task_mtl.agg({'NEXT_ACTIVITY': ['mean', 'std'], 
                                                          'NEXT_TIME': ['mean', 'std'], 
                                                          'REMAINING_TIME': ['mean', 'std'], 
                                                          'Best Epoch': ['mean', 'std'],
                                                          'Best Validation Loss': ['mean', 'std']})
                # print(df_model_task_mtl)
                # only keep the result of the learning rate with the lowest validation loss
                df_model_task_mtl = df_model_task_mtl.sort_values(by=('Best Validation Loss', 'mean'), ascending=True)
                df_model_task_mtl = df_model_task_mtl.iloc[0]
                # now make a nice dictionary from this
                best_results.append({
                    'Model': model,
                    'MTL': mtl,
                    'Tasks': task,
                    'Learning Rate': df_model_task_mtl.name[0],
                    'MTL HPO': df_model_task_mtl.name[1],
                    'NEXT_ACTIVITY_mean': df_model_task_mtl['NEXT_ACTIVITY']['mean'],
                    'NEXT_ACTIVITY_std': df_model_task_mtl['NEXT_ACTIVITY']['std'],
                    'NEXT_TIME_mean': df_model_task_mtl['NEXT_TIME']['mean'],
                    'NEXT_TIME_std': df_model_task_mtl['NEXT_TIME']['std'],
                    'REMAINING_TIME_mean': df_model_task_mtl['REMAINING_TIME']['mean'],
                    'REMAINING_TIME_std': df_model_task_mtl['REMAINING_TIME']['std'],
                    'Best Epoch_mean': df_model_task_mtl['Best Epoch']['mean'],
                })
    # print(f"Best results: {best_results}")
    return pd.DataFrame(best_results)

def main():
    parser = argparse.ArgumentParser(description='collect results')    
    parser.add_argument('--dataset', type=str, required=True, 
                        help='name of the dataset')
    args = parser.parse_args()
    root_path = os.getcwd()
    log_dir = os.path.join(root_path, 'models', args.dataset)   
    results_dir = os.path.join(root_path, 'results', args.dataset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    df, best_results_df = parse_log_files(log_dir)
    csv_path = os.path.join(results_dir, args.dataset+'_overall_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Overall results saved to {csv_path}")
    csv_path = os.path.join(results_dir, args.dataset+'_best_results.csv')
    best_results_df.to_csv(csv_path, index=False)
    print(f"Best results saved to {csv_path}")
    
if __name__ == '__main__':
    main() 