import torch
import torch.nn as nn
import math

def get_model(model_name, weighting, output_dims, model_parameters, device,
              rep_grad=False):
    if model_name == "LSTM":
        model_class = process_lstm(weighting_class=weighting)
    elif model_name == "CNN":
        model_class = process_cnn(weighting_class=weighting)
    elif model_name == "Transformer":
        model_class = process_trans(weighting_class=weighting)
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
        def __init__(self, output_dims=None, model_parameters=None,
                     device=None, rep_grad=False):
            """
            LSTM model for process prediction tasks
            
            Args:
                input_dim (int): Input feature dimension
                hidden_dim (int): Hidden layer dimension
                num_layers (int): Number of LSTM layers
                dropout (float): Dropout rate
                num_activities (int): Number of unique activities 
                                        (for next activity prediction)
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
            self.lstm = nn.LSTM(
                model_parameters["input_dim"], self.hidden_dim, self.num_layers,
                batch_first=True, dropout=model_parameters["dropout"])
            
            # Task-specific output layers
            self.task_heads = nn.ModuleDict()
            print(f"output_dims: {output_dims}")

            for task_name, dim in output_dims.items():
                if task_name != 'next_activity':
                    self.task_heads[task_name] = nn.Linear(self.hidden_dim, dim)
                else:
                    self.task_heads[task_name] = nn.Linear(
                        self.hidden_dim, model_parameters["num_activities"])
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
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, 
                dtype=torch.float32).to(self.device)
            c0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, 
                dtype=torch.float32).to(self.device)
            
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # pe: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)] # Add positional encoding
        return self.dropout(x)
    
class Trans_Encoder(nn.Module):
    def __init__(self, num_activities, num_resources, num_days, num_feat_dim,
                 emb_dim, num_heads, num_layers, dropout, pooling, max_len):
        super().__init__()
        self.num_activities = num_activities
        self.num_resources = num_resources
        self.num_days = num_days
        self.num_feat_dim = num_feat_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        # Feature processing layers
        self.embedding_cat1 = nn.Embedding(self.num_activities, self.emb_dim)
        self.embedding_cat2 = nn.Embedding(self.num_resources, self.emb_dim) 
        self.embedding_cat3 = nn.Embedding(self.num_days, 3)
        # Calculate the total input dimension for the Transformer (d_model)
        self.d_model = 2 * self.emb_dim + self.num_feat_dim + 3   
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, max_len=max_len)        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, self.num_layers
        )

    def forward(self, input_cat1, input_cat2, input_cat3, input_numerical):
        # Embed categorical features
        embedded_cat1 = self.embedding_cat1(input_cat1)
        embedded_cat2 = self.embedding_cat2(input_cat2)
        embedded_cat3 = self.embedding_cat3(input_cat3)
        # Concatenate all features
        combined_features = torch.cat(
            (embedded_cat1, embedded_cat2, embedded_cat3, input_numerical),
            dim=-1)     
        # Add positional encoding
        combined_features = self.positional_encoding(combined_features)        
        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(combined_features)  
        # transformer_output: (batch_size, seq_len, d_model)
        if self.pooling == 'mean':
            pooled_output = transformer_output.mean(dim=1)
        elif self.pooling == 'max':
            pooled_output = transformer_output.max(dim=1)
        elif self.pooling == 'min':
            pooled_output = transformer_output.min(dim=1)
        else:
            raise ValueError("Invalid pooling strategy.")         
        return pooled_output
    
def process_trans(weighting_class):
    class ProcessTrans(weighting_class):
        def __init__(self, output_dims=None, model_parameters=None,
                     device=None, rep_grad=False):
            """
            Transformer model for process prediction tasks
            
            Args:
                output_dims (dict): Dictionary of output dimensions for each task
                model_parameters (dict): Includes input_dim, hidden_dim, num_heads,
                                         num_layers, dropout, num_activities
                device (torch.device): Device to run the model on
                rep_grad (bool): Whether to store gradients for representation learning
            """
            super(ProcessTrans, self).__init__()
            self.device = device
            self.task_num = len(output_dims.keys())
            self.init_param() # initialize the parameters for the weighting method
            self.rep_grad = rep_grad
            if self.rep_grad:
                self.rep_tasks = {}
                self.rep = {}
                
            self.num_activities = model_parameters["num_activities"]
            self.num_resources = model_parameters["num_resources"]
            self.num_days = model_parameters["num_days"]
            self.num_feat_dim = model_parameters["num_feat_dim"] 
            self.emb_dim = model_parameters["emb_dim"]
            self.num_heads = model_parameters["num_heads"]
            self.num_layers = model_parameters["num_layers"]
            self.dropout = model_parameters["dropout"]
            self.pooling = model_parameters["pooling"] 
            self.d_model = 2 * self.emb_dim + self.num_feat_dim + 3
            # create Transformer encoder
            self.encoder = Trans_Encoder(
                num_activities=self.num_activities,
                num_resources=self.num_resources,
                num_days=self.num_days,
                num_feat_dim=self.num_feat_dim,
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
                pooling=self.pooling,
                max_len=model_parameters["max_len"]
            )

            # Task-specific output layers
            self.task_heads = nn.ModuleDict()
            print(f"output_dims: {output_dims}")
            for task_name, dim in output_dims.items():
                if task_name != 'next_activity':
                    self.task_heads[task_name] = nn.Linear(self.d_model, dim)
                else:
                    self.task_heads[task_name] = nn.Linear(
                        self.d_model, model_parameters["num_activities"])
            self.task_name = list(output_dims.keys())
        

        def forward(self, x):
            """
            Forward pass
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            Returns:
                dict: Dictionary of predictions for each task
            """
            # x: (batch_size, seq_len, 6) where 6 is the total number of features (3 cat, 3 num)
            # Extract categorical features (Convert to long for nn.Embedding)
            input_cat1 = x[:, :, 0].long()
            input_cat2 = x[:, :, 1].long()
            input_cat3 = x[:, :, 2].long()
            # Extract numerical features
            input_numerical = x[:, :, 3:]  # This will be (batch_size, seq_len, 3)
            encoder_output = self.encoder(
                input_cat1, input_cat2, input_cat3, input_numerical)
            
            # Update the representation and return the predictions if rep_grad is True
            if self.rep_grad:
                predictions = self.update_rep(encoder_output)
                return predictions
            
            # Generate predictions for each task
            predictions = {}
            tasks = self.task_heads.keys()
            
            for task in tasks:
                if task == 'next_activity':
                    predictions[task] = self.task_heads[task](encoder_output)
                else:
                    # For time-based predictions, ensure output is properly shaped
                    predictions[task] = self.task_heads[task](encoder_output).squeeze(-1)            
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
    return ProcessTrans

def process_cnn(weighting_class):
    class ProcessCNN(weighting_class):
        def __init__(self, output_dims=None, model_parameters=None,
                     device=None, rep_grad=False):
            """
            CNN model for process prediction tasks
            
            Args:
                input_dim (int): Input feature dimension
                hidden_dim (int): Hidden layer dimension
                num_filters (int): Number of filters in convolutional layers
                kernel_size (int): Size of convolutional kernel
                dropout (float): Dropout rate
                num_activities (int): Number of unique activities
                                    (for next activity prediction)
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

            # Calculate the correct flatten size by doing a forward pass with a dummy input
            # Use the actual max_len from the dataset
            seq_len = model_parameters.get("max_len")
            dummy_input = torch.randn(1, seq_len, model_parameters["input_dim"])
            with torch.no_grad():
                # Reshape for CNN: (batch_size, input_dim, seq_len)
                dummy_input = dummy_input.transpose(1, 2)
                # CNN forward pass
                dummy_output = self.encoder(dummy_input)
                # Flatten the output
                dummy_output = dummy_output.view(dummy_output.size(0), -1)
                actual_flatten_size = dummy_output.size(1)
            print(f"actual_flatten_size: {actual_flatten_size}")
            
            # Task-specific output layers - initialize properly like LSTM/Transformer
            self.task_heads = nn.ModuleDict()
            print(f"output_dims: {output_dims}")

            for task_name, dim in output_dims.items():
                if task_name != 'next_activity':
                    self.task_heads[task_name] = nn.Linear(actual_flatten_size, dim)
                else:
                    self.task_heads[task_name] = nn.Linear(
                        actual_flatten_size, model_parameters["num_activities"])
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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)