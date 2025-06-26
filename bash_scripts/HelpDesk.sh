#!/bin/bash
#SBATCH --job-name=mtl_helpdesk
#SBATCH --cpus-per-task=30
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/kamiriel/MTL4PPM
python main.py --data_path data/HelpDesk.csv --tasks next_activity --model Transformer --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks next_time --model Transformer --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks remaining_time --model Transformer --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks next_activity --model LSTM --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks next_time --model LSTM --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks remaining_time --model LSTM --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks next_activity --model CNN --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks next_time --model CNN --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks remaining_time --model CNN --epochs 200
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting EW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model Transformer --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model LSTM --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_activity next_time --model CNN --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model Transformer --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model LSTM --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_activity remaining_time --model CNN --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model Transformer --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model LSTM --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks next_time remaining_time --model CNN --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks multi --model Transformer --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks multi --model LSTM --epochs 200 --weighting GradDrop --rep_grad
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting PCGrad
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting GradNorm
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting CAGrad
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting IMTL
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting Nash_MTL
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting UW
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting DWA
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting GLS
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting RLW
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting UW_SO
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting UW_SO --use_softmax
python main.py --data_path data/HelpDesk.csv --tasks multi --model CNN --epochs 200 --weighting GradDrop --rep_grad