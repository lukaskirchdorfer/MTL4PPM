import argparse
import torch
from src.data.dataset import ProcessLogDataset
from src.utils.trainer import Trainer
import os
import logging

import src.weighting as weighting_method
from src.models.models import get_model, init_weights


# Configure logger
logger = logging.getLogger('MTL_PPM_Logger') 
logger.setLevel(logging.INFO) 
# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Train process prediction models')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the input CSV file')
    parser.add_argument('--max_len', type=int, default=100,
                      help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='LSTM',
                      choices=['LSTM', 'CNN', 'Transformer'],
                      help='Model architecture to use')
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='Hidden dimension size')
    parser.add_argument('--emb_dim', type=int, default=61,
                      help='Embedding size for Transformer model')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of LSTM/Transformer layers')
    parser.add_argument('--num_filters', type=int, default=64,
                      help='Number of filters in CNN layers')
    parser.add_argument('--kernel_size', type=int, default=3,
                      help='Size of CNN kernel')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of heads in Transformer')
    parser.add_argument('--pooling', type=str, default='mean',
                      choices=['mean', 'max', 'min'],
                      help='Pooling strategy for Transformer model')    
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--weighting', type=str, default='EW',
                      choices=['EW', 'UW', 'DWA', 'GLS', 'IMTL', 'RLW',
                               'CAGrad', 'GradNorm', 'GradDrop', 'PCGrad',
                               'Nash_MTL', 'UW_SO', 'Scalarization'],
                      help='Weighting strategy to use')
    
    # Task arguments
    parser.add_argument('--tasks', type=str, nargs='+', default=['next_activity'],
                      choices=['next_activity', 'next_time', 'remaining_time', 'multi'],
                      help='Prediction tasks. Use "multi" for all tasks or specify individual tasks')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience (no. epochs) for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0,
                      help='Minimum improvement observed for early stopping')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.2,
                      help='Test split ratio')
    parser.add_argument('--save_dir', type=str, default='models',
                      help='Directory to save models')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    # Weighting arguments
    parser.add_argument("--rep_grad", action="store_true", default=False, 
                        help="computing gradient for representation or sharing parameters")

    # UW-SO arguments
    parser.add_argument("--use_softmax", action="store_true", default=False, 
                        help="use softmax to normalize the weights")
    parser.add_argument("--T", type=float, default=1.0, 
                        help="temperature parameter for the softmax function")  

    # Scalarization arguments
    parser.add_argument("--scalar_weights", type=float, nargs='+', 
                        help="scalar weights for the tasks")
    
    return parser.parse_args()

def prepare_kwargs(args):
    kwargs = {}
    if args.weighting == 'DWA':
        kwargs['T'] = 2.0
    elif args.weighting == 'CAGrad':
        kwargs['calpha'] = 0.5
        kwargs['rescale'] = 1
    elif args.weighting == 'GradNorm':
        kwargs['alpha'] = 1.5
    elif args.weighting == 'GradDrop':
        kwargs['leak'] = 0.0
    elif args.weighting == 'Nash_MTL':
        kwargs['update_weights_every'] = 1
        kwargs['optim_niter'] = 20
        kwargs['max_norm'] = 1.0
    elif args.weighting == 'UW_SO':
        kwargs['T'] = args.T
        kwargs['use_softmax'] = args.use_softmax
    elif args.weighting == 'Scalarization':
        kwargs['scalar_weights'] = args.scalar_weights
    return kwargs

def main():
    # Parse arguments
    args = parse_args()
    kwargs = prepare_kwargs(args)

    print(f"Device: {args.device}")
    
    # If 'multi' is specified, use all tasks
    tasks = ['next_activity', 'next_time', 'remaining_time'] if 'multi' in args.tasks else args.tasks
    
    # Create save directory if it doesn't exist
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    args.save_dir = os.path.join(args.save_dir, dataset_name)    
    os.makedirs(args.save_dir, exist_ok=True)
    # Name of the MTL approach for saving results and reports
    if args.weighting == 'UW_SO' and not args.use_softmax:
        mtl_name = 'UW_O'
    else:
        mtl_name = args.weighting      
    
    # set the logger to report important results
    logger_path = os.path.join(
        args.save_dir, f'{args.model}_{"_".join(tasks)}_{mtl_name}.log')   
    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Load dataset
    print(f'Loading data from {args.data_path}...')
    logger.info(f'Loading data from {args.data_path}...') 

    # Create the full dataset first
    full_dataset = ProcessLogDataset(
        csv_path=args.data_path,
        tasks=tasks  
    )
    
    # Split into train and validation sets
    train_val_size = int((1 - args.test_split) * len(full_dataset))
    train_size = int((1 - args.val_split) * train_val_size)    
    val_size = train_val_size - train_size
    test_size = len(full_dataset) - train_val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]) 
    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')
    logger.info(f'Train size: {len(train_dataset)}')
    logger.info(f'Validation size: {len(val_dataset)}')
    logger.info(f'Test size: {len(test_dataset)}')
    
    # Create model
    print('Initializing model...')
    logger.info('Initializing model...')
    output_dims = {}
    if len(tasks) > 1:  # Multi-task case
        if 'next_activity' in tasks:
            output_dims['next_activity'] = full_dataset.num_activities
        if 'next_time' in tasks:
            output_dims['next_time'] = 1
        if 'remaining_time' in tasks:
            output_dims['remaining_time'] = 1
    else:  # Single task case
        if tasks[0] != 'next_activity':
            output_dims = {tasks[0]: 1}
        else:
            output_dims = {'next_activity': full_dataset.num_activities}

    weighting = weighting_method.__dict__[args.weighting]

    if args.model == 'LSTM':
        model_parameters = {
            "input_dim": full_dataset.feature_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "num_activities": full_dataset.num_activities,
        }
    elif args.model == 'CNN':
        model_parameters = {
            "input_dim": full_dataset.feature_dim,
            "hidden_dim": args.hidden_dim,
            "num_filters": args.num_filters,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "num_activities": full_dataset.num_activities,
            "max_len": full_dataset.max_len,
        }
    elif args.model == 'Transformer':
        model_parameters = {
            "num_feat_dim": full_dataset.num_feat_dim,
            "emb_dim": args.emb_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "num_activities": full_dataset.num_activities,
            "num_resources": full_dataset.num_resources, 
            "num_days": full_dataset.num_days,
            "pooling": args.pooling,
        }
    else:
        raise ValueError(f"Model {args.model} not found")
    
    model = get_model(
        model_name=args.model,
        weighting=weighting,
        output_dims=output_dims,
        model_parameters=model_parameters,
        device=args.device,
        rep_grad=args.rep_grad,
    )
    
    model.apply(init_weights)  
    if args.model == 'Transformer':
        model = model.double()
        
    # Initialize trainer with the full dataset
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        weighting=args.weighting,
        patience=args.patience,
        min_delta= args.min_delta,
        logger=logger,
        **kwargs
    )
    
    # Train model
    print('Starting training...')
    logger.info('Starting training...')
    save_path = os.path.join(
        args.save_dir,
        f'{args.model}_{"_".join(tasks)}_{mtl_name}_model.pt')
    weight_path = os.path.join(
        args.save_dir,
        f'{args.model}_{"_".join(tasks)}_{mtl_name}_task_weights.json')
    trainer.train(args.epochs, save_path, weight_path)
    
    
    # Inference on test dataset 
    task_names = '_'.join(tasks)
    if len(tasks) > 1:
        learn_title = '_MTL_'+task_names+'_task_'
    else:
        learn_title = '_STL_'+task_names+'_task_'                
    inference_name_lst = [
        f"{args.model}_{learn_title}_{task}_{mtl_name}_.csv" 
        for task in tasks]
    inference_path_lst = [os.path.join(args.save_dir, name) 
                          for name in inference_name_lst]
    trainer.inference(inference_path_lst)
    
if __name__ == '__main__':
    main() 