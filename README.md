# Deep Learning for Predictive Process Monitoring

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
│   └── utils/             # Utility functions and helper classes
├── config/                # Configuration files
└── main.py               # Main training script
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script accepts various arguments to configure the training process:

```bash
python main.py \
    --data_path data/your_log.csv \
    --task next_activity \  # Options: next_activity, next_time, remaining_time, multi
    --model lstm \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001
```

## Input Data Format

The input data should be a CSV file with the following columns:
- case_id: Identifier for the process instance
- activity: Name of the activity
- timestamp: Timestamp of the activity execution
- resource: Resource who performed the activity 