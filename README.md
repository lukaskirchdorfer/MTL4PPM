# Multi-Task Deep Learning for Predictive Process Monitoring

This is the supplementary GitHub repository of the paper: "On the Potential of Multi-Task Learning in Predictive Process Monitoring" by Lukas Kirchdorfer, Keyvan Amiri Elyasi and Heiner Stuckenschmidt.

We provide supplementary material to the main paper in Appendix.pdf.

This repository contains PyTorch implementations of deep learning models for predictive process monitoring tasks, including:
- Next Activity Prediction
- Next Time Prediction
- Remaining Time Prediction

## Project Structure
```
.
├── data/                   # Directory for storing input data
├── src/
│   ├── models/            # Neural network model implementations
│   ├── data/              # Data processing and dataset classes
│   └── utils/             # Utility functions and helper classes including the trainer class
│   └── weighting/         # MTL loss and gradient weighting classes
├── MTL4PPM_Supplementary_Material.pdf # Supplementary material for detailed of experimental setup and results 
└── main.py               # Main training script
```

## Setup

1. Create a virtual environment:
```bash
conda create -n mtl4ppm python=3.9
conda activate mtl4ppm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script accepts various arguments to configure the training process:

```bash
python main.py \
    --data_path data/P2P.csv \ # Upload your event log to the data folder
    --tasks next_activity next_time \  # Options: next_activity, next_time, remaining_time, multi
    --model LSTM \  # Options: LSTM, CNN, Transformer
    --epochs 200 \
    --weighting UW \ # Options: see /source/weighting/
```

## Input Data Format

The input data should be a CSV file with the following columns:
- case_id: Identifier for the process instance
- activity: Name of the activity
- timestamp: Timestamp of the activity execution
- resource: Resource who performed the activity 
