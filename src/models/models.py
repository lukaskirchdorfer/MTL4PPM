import torch
import torch.nn as nn

def get_model(model_name, weighting, output_dims, model_parameters, device, rep_grad=False):
    if model_name == "LSTM":
        model_class = process_lstm(weighting_class=weighting)
    elif model_name == "CNN":
        model_class = process_cnn(weighting_class=weighting)
    else:
        raise ValueError(f"Model {model_name} not found")
    
    model = model_class(
        output_dims=output_dims,
        model_parameters=model_parameters,
        device=device,
        rep_grad=rep_grad,
    )
    
    return model

def process_lstm(weighting_class):
    class ProcessLSTM(weighting_class):
        def __init__(self, output_dims=None, model_parameters=None, device=None, rep_grad=False):
            """
            LSTM model for process prediction tasks
            
            Args:
                input_dim (int): Input feature dimension
                hidden_dim (int): Hidden layer dimension
                num_layers (int): Number of LSTM layers
                dropout (float): Dropout rate
                num_activities (int): Number of unique activities (for next activity prediction)
                output_dims (dict): Dictionary of output dimensions for each task
                                e.g., {'next_time': 1, 'remaining_time': 1}
            """
            # Initialize the weighting class first with the device
            super(ProcessLSTM, self).__init__()
            self.device = device
            self.task_num = len(output_dims.keys())
            self.init_param() # initialize the parameters for the weighting method
            self.rep_grad = rep_grad
            if self.rep_grad:
                self.rep_tasks = {}
                self.rep = {}

            self.hidden_dim = model_parameters["hidden_dim"]
            self.num_layers = model_parameters["num_layers"]
            self.lstm = nn.LSTM(model_parameters["input_dim"], self.hidden_dim, self.num_layers, 
                            batch_first=True, dropout=model_parameters["dropout"])
            
            # Task-specific output layers
            self.task_heads = nn.ModuleDict()
            print(f"output_dims: {output_dims}")

            for task_name, dim in output_dims.items():
                if task_name != 'next_activity':
                    self.task_heads[task_name] = nn.Linear(self.hidden_dim, dim)
                else:
                    self.task_heads[task_name] = nn.Linear(self.hidden_dim, model_parameters["num_activities"])
            self.task_name = list(output_dims.keys())
            
            
        def forward(self, x):
            """
            Forward pass
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            Returns:
                dict: Dictionary of predictions for each task
            """
            batch_size = x.size(0)
            
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, dtype=torch.float32).to(self.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, dtype=torch.float32).to(self.device)
            
            # LSTM forward pass
            lstm_out, _ = self.lstm(x, (h0, c0))
            
            # Get the output for the last time step
            last_output = lstm_out[:, -1, :]

            # Update the representation and return the predictions if rep_grad is True
            if self.rep_grad:
                predictions = self.update_rep(last_output)
                return predictions
            
            # Generate predictions for each task
            predictions = {}
            tasks = self.task_heads.keys()
            
            for task in tasks:
                if task == 'next_activity':
                    predictions[task] = self.task_heads[task](last_output)
                else:
                    # For time-based predictions, ensure output is properly shaped
                    predictions[task] = self.task_heads[task](last_output).squeeze(-1)
            
            return predictions
        
        def get_share_params(self):
            r"""Return the parameters of the encoder part of the model.
            """
            return self.lstm.parameters()
        
        def zero_grad_share_params(self):
            r"""Set gradients of the shared parameters to zero.
            """
            self.lstm.zero_grad()

        def update_rep(self, s_rep):
            predictions = {}
            same_rep = True if not isinstance(s_rep, list) else False
            for tn, task in enumerate(self.task_name):
                ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
                ss_rep = self._prepare_rep(ss_rep, task, same_rep)
                predictions[task] = self.task_heads[task](ss_rep)
            return predictions

        def _prepare_rep(self, rep, task, same_rep=None):
            if self.rep_grad:
                if not same_rep:
                    self.rep[task] = rep
                else:
                    self.rep = rep
                self.rep_tasks[task] = rep.detach().clone()
                self.rep_tasks[task].requires_grad = True
                return self.rep_tasks[task]
            else:
                return rep
        
    return ProcessLSTM

def process_cnn(weighting_class):
    class ProcessCNN(weighting_class):
        def __init__(self, output_dims=None, model_parameters=None, device=None, rep_grad=False):
            """
            CNN model for process prediction tasks
            
            Args:
                input_dim (int): Input feature dimension
                hidden_dim (int): Hidden layer dimension
                num_filters (int): Number of filters in convolutional layers
                kernel_size (int): Size of convolutional kernel
                dropout (float): Dropout rate
                num_activities (int): Number of unique activities (for next activity prediction)
                output_dims (dict): Dictionary of output dimensions for each task
                                e.g., {'next_time': 1, 'remaining_time': 1}
            """
            # Initialize the weighting class first with the device
            super(ProcessCNN, self).__init__()
            self.device = device
            self.output_dims = output_dims
            self.model_parameters = model_parameters
            self.task_num = len(output_dims.keys())
            self.init_param() # initialize the parameters for the weighting method
            self.rep_grad = rep_grad
            if self.rep_grad:
                self.rep_tasks = {}
                self.rep = {}

            self.hidden_dim = model_parameters["hidden_dim"]
            self.num_filters = model_parameters["num_filters"]
            self.kernel_size = model_parameters["kernel_size"]
            
            self.encoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=model_parameters["input_dim"],
                    out_channels=self.num_filters,
                    kernel_size=self.kernel_size,
                    padding='same'
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(model_parameters["dropout"]),

                nn.Conv1d(
                    in_channels=self.num_filters,
                    out_channels=self.num_filters * 2,
                    kernel_size=self.kernel_size,
                    padding='same'
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(model_parameters["dropout"]),

                nn.Conv1d(
                    in_channels=self.num_filters * 2,
                    out_channels=self.num_filters * 4,
                    kernel_size=self.kernel_size,
                    padding='same'
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(model_parameters["dropout"]),
            )

            
            # We'll calculate the flatten size in the first forward pass
            self.flatten_size = None
            
            # Task-specific output layers
            self.task_heads = nn.ModuleDict()

            for task_name, dim in output_dims.items():
                if task_name != 'next_activity':
                    self.task_heads[task_name] = None  # Will be initialized in first forward pass
                else:
                    self.task_heads[task_name] = None  # Will be initialized in first forward pass
            self.task_name = list(output_dims.keys())
            
        def forward(self, x):
            """
            Forward pass
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            Returns:
                dict: Dictionary of predictions for each task
            """
            # Reshape input for CNN: (batch_size, input_dim, seq_len)
            x = x.transpose(1, 2)
            
            # CNN forward pass
            x = self.encoder(x)
            
            # Flatten the output
            x = x.view(x.size(0), -1)
            
            # Initialize task heads if this is the first forward pass
            if self.flatten_size is None:
                self.flatten_size = x.size(1)
                for task_name, dim in self.output_dims.items():
                    if task_name != 'next_activity':
                        self.task_heads[task_name] = nn.Linear(self.flatten_size, dim).to(self.device)
                    else:
                        self.task_heads[task_name] = nn.Linear(self.flatten_size, self.model_parameters["num_activities"]).to(self.device)
            
            # Update the representation and return the predictions if rep_grad is True
            if self.rep_grad:
                predictions = self.update_rep(x)
                return predictions
            
            # Generate predictions for each task
            predictions = {}
            tasks = self.task_heads.keys()
            
            for task in tasks:
                if task == 'next_activity':
                    predictions[task] = self.task_heads[task](x)
                else:
                    # For time-based predictions, ensure output is properly shaped
                    predictions[task] = self.task_heads[task](x).squeeze(-1)
            
            return predictions
        
        def get_share_params(self):
            r"""Return the parameters of the encoder part of the model.
            """
            return self.encoder.parameters()
        
        def zero_grad_share_params(self):
            r"""Set gradients of the shared parameters to zero.
            """
            self.encoder.zero_grad()

        def update_rep(self, s_rep):
            predictions = {}
            same_rep = True if not isinstance(s_rep, list) else False
            for tn, task in enumerate(self.task_name):
                ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
                ss_rep = self._prepare_rep(ss_rep, task, same_rep)
                predictions[task] = self.task_heads[task](ss_rep)
            return predictions

        def _prepare_rep(self, rep, task, same_rep=None):
            if self.rep_grad:
                if not same_rep:
                    self.rep[task] = rep
                else:
                    self.rep = rep
                self.rep_tasks[task] = rep.detach().clone()
                self.rep_tasks[task].requires_grad = True
                return self.rep_tasks[task]
            else:
                return rep
        
    return ProcessCNN