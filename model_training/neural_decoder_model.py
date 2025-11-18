"""
PyTorch port of NeuralDecoder's TensorFlow GRU model.
Original: NeuralDecoder/neuralDecoder/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDecoder_NeuralDecoder(nn.Module):
    """
    PyTorch port of the NeuralDecoder GRU model from TensorFlow.
    
    This matches the architecture in NeuralDecoder/neuralDecoder/models.py
    """
    def __init__(
        self,
        units,
        n_classes,
        weight_reg=1e-5,
        act_reg=0.0,
        subsample_factor=1,
        bidirectional=False,
        dropout=0.0,
        n_layers=2,
        conv_kwargs=None,
        stack_kwargs=None,
    ):
        """
        Args:
            units (int): Number of hidden units in each GRU layer
            n_classes (int): Number of output classes (for CTC, this is diphones + blank)
            weight_reg (float): L2 regularization strength (not used in forward, handled by optimizer)
            act_reg (float): Activity regularization (not implemented)
            subsample_factor (int): Factor to subsample RNN outputs (applied after 2nd-to-last layer)
            bidirectional (bool): Whether to use bidirectional GRU
            dropout (float): Dropout rate for GRU layers
            n_layers (int): Number of GRU layers
            conv_kwargs (dict): Arguments for optional DepthwiseConv1D (e.g., {'kernel_size': 3})
            stack_kwargs (dict): Arguments for input patching (e.g., {'kernel_size': 14, 'strides': 4})
        """
        super().__init__()
        
        self.units = units
        self.n_classes = n_classes
        self.subsample_factor = subsample_factor
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.stack_kwargs = stack_kwargs
        self.weight_reg = weight_reg
        
        # Calculate output size (doubled if bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.output_size = units * self.num_directions
        
        # Initial states (learnable)
        if bidirectional:
            # For bidirectional: 2 states per layer (forward, backward)
            self.init_states = nn.ParameterList([
                nn.Parameter(torch.randn(1, units) * 0.1),
                nn.Parameter(torch.randn(1, units) * 0.1),
            ])
        else:
            self.init_states = nn.Parameter(torch.randn(1, units) * 0.1)
        
        # Optional depthwise convolution (1D smoothing)
        self.conv1 = None
        if conv_kwargs is not None:
            # PyTorch doesn't have DepthwiseConv1D, so we use Conv1d with groups=in_channels
            # This will be set after first forward pass when we know input size
            self.conv_kwargs = conv_kwargs
        
        # GRU layers
        # Note: input_size for first layer will be set when the full model is created
        # For now, we create layers with default input size = units
        self.rnn_layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            input_size = units  # First layer input will be overridden if needed
            rnn = nn.GRU(
                input_size=input_size,
                hidden_size=units,
                num_layers=1,
                batch_first=True,
                dropout=0.0,  # We handle dropout manually
                bidirectional=bidirectional
            )
            self.rnn_layers.append(rnn)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output dense layer
        self.dense = nn.Linear(self.output_size, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with orthogonal for recurrent, glorot for feedforward."""
        for rnn in self.rnn_layers:
            for name, param in rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
    
    def forward(self, x, states=None, return_state=False):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, features)
            states (list of Tensors): Hidden states for each layer. If None, uses learned init states.
            return_state (bool): Whether to return hidden states along with output
            
        Returns:
            logits (Tensor): Output logits of shape (batch_size, time_steps, n_classes)
            new_states (list of Tensors): Hidden states after processing (if return_state=True)
        """
        batch_size = x.size(0)
        
        # Apply input patching/stacking if configured
        if self.stack_kwargs is not None:
            x = self._apply_patching(x)
        
        # Apply optional depthwise convolution
        if self.conv1 is not None:
            x = self.conv1(x)
            x = F.relu(x)
        
        # Initialize states if not provided
        if states is None:
            states = self._get_initial_states(batch_size)
        
        # Pass through RNN layers
        new_states = []
        for layer_idx, rnn in enumerate(self.rnn_layers):
            # Get state for this layer
            if self.bidirectional:
                # State shape: (2, batch_size, hidden_size) for bidirectional
                state = states[layer_idx]
            else:
                # State shape: (1, batch_size, hidden_size) for unidirectional
                state = states[layer_idx].unsqueeze(0) if states[layer_idx].dim() == 2 else states[layer_idx]
            
            # Forward through RNN
            x, h = rnn(x, state)
            
            # Apply subsampling after 2nd-to-last layer
            if layer_idx == self.n_layers - 2 and self.subsample_factor > 1:
                x = x[:, ::self.subsample_factor, :]
            
            # Apply dropout (except on last layer output)
            if self.dropout_layer is not None and layer_idx < self.n_layers - 1:
                x = self.dropout_layer(x)
            
            # Store new state
            new_states.append(h)
        
        # Output layer
        logits = self.dense(x)
        
        if return_state:
            return logits, new_states
        else:
            return logits
    
    def _apply_patching(self, x):
        """
        Apply input patching (stacking) to compress temporal dimension.
        
        Equivalent to TensorFlow's tf.image.extract_patches.
        
        Args:
            x (Tensor): Input of shape (batch_size, time_steps, features)
            
        Returns:
            Tensor: Patched input of shape (batch_size, num_patches, patch_size * features)
        """
        kernel_size = self.stack_kwargs['kernel_size']
        stride = self.stack_kwargs['strides']
        
        # Add dummy dimension for unfold: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        
        # Extract patches using unfold
        # unfold(dimension, size, step)
        x_unfold = x.unfold(2, kernel_size, stride)  # (B, F, num_patches, kernel_size)
        
        # Rearrange: (B, F, num_patches, kernel_size) -> (B, num_patches, kernel_size, F)
        x_unfold = x_unfold.permute(0, 2, 3, 1)
        
        # Flatten last two dimensions: (B, num_patches, kernel_size * F)
        batch_size, num_patches, kernel_size, features = x_unfold.shape
        x = x_unfold.reshape(batch_size, num_patches, kernel_size * features)
        
        return x
    
    def _get_initial_states(self, batch_size):
        """Get initial hidden states, expanded to batch size."""
        states = []
        
        if self.bidirectional:
            # For bidirectional: each layer needs (2, batch, hidden)
            for _ in range(self.n_layers):
                forward_state = self.init_states[0].expand(batch_size, -1)
                backward_state = self.init_states[1].expand(batch_size, -1)
                # Stack to (2, batch, hidden)
                state = torch.stack([forward_state, backward_state], dim=0)
                states.append(state)
        else:
            # For unidirectional: each layer needs (1, batch, hidden)
            for _ in range(self.n_layers):
                state = self.init_states.expand(batch_size, -1)
                states.append(state)
        
        return states
    
    def get_subsampled_timesteps(self, timesteps):
        """
        Calculate the number of timesteps after patching and subsampling.
        
        Args:
            timesteps (int or Tensor): Original number of timesteps
            
        Returns:
            int or Tensor: Number of timesteps after processing
        """
        if isinstance(timesteps, int):
            result = timesteps // self.subsample_factor
            if self.stack_kwargs is not None:
                kernel_size = self.stack_kwargs['kernel_size']
                stride = self.stack_kwargs['strides']
                result = (result - kernel_size) // stride + 1
            return result
        else:
            # Tensor version
            result = timesteps // self.subsample_factor
            if self.stack_kwargs is not None:
                kernel_size = self.stack_kwargs['kernel_size']
                stride = self.stack_kwargs['strides']
                result = (result - kernel_size) // stride + 1
            return result.int()


class GRUDecoder_NeuralDecoder_WithInputNet(nn.Module):
    """
    Full model with day-specific input networks + GRU decoder.
    
    This combines the input network preprocessing with the GRU model,
    matching the full NeuralDecoder pipeline.
    """
    def __init__(
        self,
        neural_dim,
        n_days,
        input_layer_sizes,
        gru_units,
        n_classes,
        n_gru_layers=5,
        input_activation='softsign',
        input_dropout=0.2,
        rnn_dropout=0.4,
        weight_reg=1e-5,
        bidirectional=False,
        stack_kwargs=None,
    ):
        """
        Args:
            neural_dim (int): Number of input features (e.g., 512)
            n_days (int): Number of recording days (for day-specific layers)
            input_layer_sizes (list): Sizes for day-specific layers (e.g., [256])
            gru_units (int): Hidden units in GRU (e.g., 512)
            n_classes (int): Number of output classes
            n_gru_layers (int): Number of GRU layers
            input_activation (str): Activation for input layers ('softsign', 'relu', 'tanh')
            input_dropout (float): Dropout rate for input layers
            rnn_dropout (float): Dropout rate for GRU
            weight_reg (float): L2 regularization strength
            bidirectional (bool): Bidirectional GRU
            stack_kwargs (dict): Patching configuration
        """
        super().__init__()
        
        self.neural_dim = neural_dim
        self.n_days = n_days
        self.input_layer_sizes = input_layer_sizes
        
        # Day-specific input networks
        self.day_networks = nn.ModuleList()
        for day_idx in range(n_days):
            layers = []
            in_features = neural_dim
            
            for layer_idx, out_features in enumerate(input_layer_sizes):
                # Use identity init if dimensions match
                linear = nn.Linear(in_features, out_features)
                if in_features == out_features:
                    nn.init.eye_(linear.weight)
                else:
                    nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                
                layers.append(linear)
                
                # Activation
                if input_activation == 'softsign':
                    layers.append(nn.Softsign())
                elif input_activation == 'relu':
                    layers.append(nn.ReLU())
                elif input_activation == 'tanh':
                    layers.append(nn.Tanh())
                
                # Dropout
                if input_dropout > 0:
                    layers.append(nn.Dropout(input_dropout))
                
                in_features = out_features
            
            self.day_networks.append(nn.Sequential(*layers))
        
        # GRU decoder
        # Need to handle input size for first GRU layer based on patching
        gru_input_size = input_layer_sizes[-1]
        if stack_kwargs is not None:
            gru_input_size = input_layer_sizes[-1] * stack_kwargs['kernel_size']
        
        # Manually set input size for first layer
        self.gru_decoder = GRUDecoder_NeuralDecoder(
            units=gru_units,
            n_classes=n_classes,
            weight_reg=weight_reg,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
            n_layers=n_gru_layers,
            stack_kwargs=stack_kwargs,
        )
        
        # Fix input size for first GRU layer
        self.gru_decoder.rnn_layers[0] = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=bidirectional
        )
        # Re-initialize
        for name, param in self.gru_decoder.rnn_layers[0].named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, day_idx, states=None, return_state=False):
        """
        Forward pass with day-specific input processing.
        
        Args:
            x (Tensor): Input of shape (batch_size, time_steps, neural_dim)
            day_idx (Tensor): Day indices of shape (batch_size,)
            states: Hidden states for GRU
            return_state (bool): Whether to return states
            
        Returns:
            logits (Tensor): Output logits
            states (optional): Hidden states if return_state=True
        """
        batch_size = x.size(0)
        
        # Apply day-specific input networks
        # Process each sample with its corresponding day network
        x_processed = []
        for i in range(batch_size):
            day_i = day_idx[i].item()
            x_i = x[i:i+1]  # Keep batch dimension
            x_i = self.day_networks[day_i](x_i)
            x_processed.append(x_i)
        x = torch.cat(x_processed, dim=0)
        
        # Pass through GRU decoder
        return self.gru_decoder(x, states=states, return_state=return_state)

