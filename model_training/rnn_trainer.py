import torch 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle

from dataset import BrainToTextDataset, train_test_split_indicies
from data_augmentations import gauss_smooth
from diphone_utils import marginalize_diphone_probabilities
import torch.nn.functional as F_torch

import torchaudio.functional as F # for edit distance
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high') # makes float32 matmuls faster on some GPUs
torch.backends.cudnn.deterministic = True # makes training more reproducible
torch._dynamo.config.cache_size_limit = 64

from rnn_model import GRUDecoder

class BrainToTextDecoder_Trainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder
    
    Written by Nick Card and Zachery Fogg with reference to Stanford NPTL's decoding function
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 

        self.best_val_PER = torch.inf # track best PER for checkpointing
        self.best_val_loss = torch.inf # track best loss for checkpointing

        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 

        self.transform_args = self.args['dataset']['data_transforms']

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=False)

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']: 
            os.makedirs(self.args['checkpoint_dir'], exist_ok=False)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device pytorch will use 
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        # Determine best AMP dtype based on GPU capability
        if torch.cuda.is_available() and self.device.type == 'cuda':
            # Check if bfloat16 is supported (A100, A10G, H100, etc.)
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.logger.info('Using bfloat16 for automatic mixed precision')
            else:
                # T4, V100, etc. only support float16
                self.amp_dtype = torch.float16
                self.logger.info('Using float16 for automatic mixed precision (bfloat16 not supported on this GPU)')
        else:
            self.amp_dtype = torch.float32
            self.logger.info('Using float32 (no AMP on CPU)')

        # Set seed if provided 
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.args['seed'])  # Set CUDA seed for all GPUs

        # Initialize the model 
        self.model = GRUDecoder(
            neural_dim = self.args['model']['n_input_features'],
            n_units = self.args['model']['n_units'],
            n_days = len(self.args['dataset']['sessions']),
            n_classes  = self.args['dataset']['n_classes'],
            rnn_dropout = self.args['model']['rnn_dropout'], 
            input_dropout = self.args['model']['input_network']['input_layer_dropout'], 
            n_layers = self.args['model']['n_layers'],
            patch_size = self.args['model']['patch_size'],
            patch_stride = self.args['model']['patch_stride'],
        )

        # Call torch.compile to speed up training
        self.logger.info("Using torch.compile")
        self.model = torch.compile(self.model)

        self.logger.info(f"Initialized RNN decoding model")

        self.logger.info(self.model)

        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        # Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
        
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")

        # Create datasets and dataloaders
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]

        # Ensure that there are no duplicate days
        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions listed in the val dataset")

        # Split trials into train and test sets
        train_trials, _ = train_test_split_indicies(
            file_paths = train_file_paths, 
            test_percentage = 0,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )
        _, val_trials = train_test_split_indicies(
            file_paths = val_file_paths, 
            test_percentage = 1,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )

        # Save dictionaries to output directory to know which trials were train vs val 
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            json.dump({'train' : train_trials, 'val': val_trials}, f)

        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None: 
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using only a subset of features: {feature_subset}')
            
        # train dataset and dataloader
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials,
            split = 'train',
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = self.args['dataset']['loader_shuffle'],
            num_workers = self.args['dataset']['num_dataloader_workers'],
            pin_memory = True 
        )

        # val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, 
            split = 'test',
            days_per_batch = None,
            n_batches = None,
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset   
            )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = False, 
            num_workers = 0,
            pin_memory = True 
        )

        self.logger.info("Successfully initialized datasets")

        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min'] / self.args['lr_max'],
                total_iters = self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")
        
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)
        
        # Log composite loss configuration
        if self.args.get('use_composite_loss', False):
            if self.args.get('use_alpha_schedule', False):
                # Using dynamic alpha schedule
                start = self.args.get('alpha_schedule_start', 0.0)
                end = self.args.get('alpha_schedule_end', 0.6)
                step = self.args.get('alpha_schedule_step_size', 0.1)
                interval = self.args.get('alpha_schedule_step_interval', 3000)
                self.logger.info(f'Using composite loss with dynamic alpha schedule:')
                self.logger.info(f'  α starts at {start:.2f}, increases by {step:.2f} every {interval} batches, caps at {end:.2f}')
                self.logger.info(f'  Loss: L = α*(phoneme) + (1-α)*(diphone)')
            else:
                # Using fixed alpha
                alpha = self.args.get('composite_loss_alpha', 0.5)
                self.logger.info(f'Using composite loss with fixed α = {alpha:.2f}')
                self.logger.info(f'  Loss: L = {alpha:.2f}*(phoneme) + {1-alpha:.2f}*(diphone)')
        else:
            self.logger.info('Using standard diphone-only CTC loss')

        # If a checkpoint is provided, then load from checkpoint
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Set rnn and/or input layers to not trainable if specified 
        for name, param in self.model.named_parameters():
            if not self.args['model']['rnn_trainable'] and 'gru' in name:
                param.requires_grad = False

            elif not self.args['model']['input_network']['input_trainable'] and 'day' in name:
                param.requires_grad = False

        # Send model to device 
        self.model.to(self.device)
        
    def get_composite_loss_alpha(self, batch_idx):
        """
        Calculate the current alpha value for composite loss based on batch index.
        
        If use_alpha_schedule is enabled, alpha increases linearly over time:
        - Starts at alpha_schedule_start
        - Increases by alpha_schedule_step_size every alpha_schedule_step_interval batches
        - Caps at alpha_schedule_end
        
        Example with default settings (start=0.0, end=0.6, step=0.1, interval=3000):
        - Batch 0-2999: alpha = 0.0 (diphone-only)
        - Batch 3000-5999: alpha = 0.1
        - Batch 6000-8999: alpha = 0.2
        - Batch 9000-11999: alpha = 0.3
        - Batch 12000-14999: alpha = 0.4
        - Batch 15000-17999: alpha = 0.5
        - Batch 18000+: alpha = 0.6 (capped)
        
        Args:
            batch_idx: Current training batch index
            
        Returns:
            alpha: Weight for phoneme loss (0.0 to 1.0)
        """
        if not self.args.get('use_alpha_schedule', False):
            # Use fixed alpha
            return self.args.get('composite_loss_alpha', 0.5)
        
        # Dynamic alpha schedule
        start_alpha = self.args.get('alpha_schedule_start', 0.0)
        end_alpha = self.args.get('alpha_schedule_end', 0.6)
        step_size = self.args.get('alpha_schedule_step_size', 0.1)
        step_interval = self.args.get('alpha_schedule_step_interval', 3000)
        
        # Calculate how many steps have occurred
        num_steps = batch_idx // step_interval
        
        # Calculate current alpha
        current_alpha = start_alpha + (num_steps * step_size)
        
        # Cap at end_alpha
        current_alpha = min(current_alpha, end_alpha)
        
        return current_alpha
        
    def marginalize_diphone_logits_to_phoneme_logits(self, diphone_logits):
        """
        Convert diphone logits to phoneme logits using log-sum-exp marginalization.
        
        This is the PyTorch/logit-space version of marginalize_diphone_probabilities()
        from diphone_utils.py. Both use the same marginalization logic (summing all
        diphones ending in the same phoneme), but this version:
        - Works on logits (before softmax) for numerical stability during training
        - Uses log-sum-exp instead of regular sum
        - Operates in PyTorch on GPU
        
        Marginalization logic:
        - Diphone encoding: prev_phoneme * 41 + curr_phoneme
        - For each phoneme i, sum all diphones where (diphone_id % 41) == i
        - This is equivalent to: diphone_probs[:, :, i::41].sum(axis=-1)
        
        Args:
            diphone_logits: (batch, time, 1681) - logits for diphone classes
            
        Returns:
            phoneme_logits: (batch, time, 41) - logits for phoneme classes
        """
        batch_size, time_steps, _ = diphone_logits.shape
        num_phonemes = 41
        
        # Initialize phoneme logits
        phoneme_logits = torch.zeros((batch_size, time_steps, num_phonemes), 
                                     device=diphone_logits.device, 
                                     dtype=diphone_logits.dtype)
        
        # For each phoneme, collect all diphone logits that end in that phoneme
        # Using the same indexing as marginalize_diphone_probabilities: i::41
        for i in range(num_phonemes):
            # Get all diphones ending in phoneme i (indices: i, i+41, i+82, ...)
            diphone_indices = list(range(i, 1681, num_phonemes))
            
            # Get logits for all diphones ending in phoneme i
            relevant_logits = diphone_logits[:, :, diphone_indices]  # (batch, time, 41)
            
            # Log-sum-exp to marginalize in log space: log(sum(exp(logit_j)))
            # This is numerically stable equivalent of: log(sum(probs ending in phoneme i))
            phoneme_logits[:, :, i] = torch.logsumexp(relevant_logits, dim=-1)
        
        return phoneme_logits

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups 

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate
        '''
        bias_params = [p for name, p in self.model.named_parameters() if 'gru.bias' in name or 'out.bias' in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name]

        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else: 
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
            
        # Fused optimizer requires params to be on CUDA
        # Since optimizer is created before model.to(device), params are still on CPU
        # So we check if any parameter is already on CUDA
        params_on_cuda = any(p.is_cuda for p in self.model.parameters())
        use_fused = params_on_cuda and torch.cuda.is_available()
        
        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = use_fused  # Only use fused when params are on CUDA
        )

        return optim 

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            '''
            Create lr lambdas for each param group that implement cosine decay

            Different lr lambda decaying for day params vs rest of the model
            '''
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_lr_ratio
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            
            # After cosine decay is complete, maintain min_lr_ratio
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min_day / lr_max_day, 
                    lr_decay_steps_day,
                    lr_warmup_steps_day, 
                    ), # day params
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")
        
        return LambdaLR(optim, lr_lambdas, -1)
        
    def load_model_checkpoint(self, load_path):
        ''' 
        Load a training checkpoint
        '''
        checkpoint = torch.load(load_path, weights_only = False) # checkpoint is just a dict

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER'] # best phoneme error rate
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf

        self.model.to(self.device)
        
        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss):
        '''
        Save a training checkpoint
        '''

        checkpoint = {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.learning_rate_scheduler.state_dict(),
            'val_PER' : PER,
            'val_loss' : loss
        }
        
        torch.save(checkpoint, save_path)
        
        self.logger.info("Saved model to checkpoint: " + save_path)

        # Save the args file alongside the checkpoint
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    def create_attention_mask(self, sequence_lengths):

        max_length = torch.max(sequence_lengths).item()

        batch_size = sequence_lengths.size(0)
        
        # Create a mask for valid key positions (columns)
        # Shape: [batch_size, max_length]
        key_mask = torch.arange(max_length, device=sequence_lengths.device).expand(batch_size, max_length)
        key_mask = key_mask < sequence_lengths.unsqueeze(1)
        
        # Expand key_mask to [batch_size, 1, 1, max_length]
        # This will be broadcast across all query positions
        key_mask = key_mask.unsqueeze(1).unsqueeze(1)
        
        # Create the attention mask of shape [batch_size, 1, max_length, max_length]
        # by broadcasting key_mask across all query positions
        attention_mask = key_mask.expand(batch_size, 1, max_length, max_length)
        
        # Convert boolean mask to float mask:
        # - True (valid key positions) -> 0.0 (no change to attention scores)
        # - False (padding key positions) -> -inf (will become 0 after softmax)
        attention_mask_float = torch.where(attention_mask, 
                                        True,
                                        False)
        
        return attention_mask_float

    def transform_data(self, features, n_time_steps, mode = 'train'):
        '''
        Apply various augmentations and smoothing to data
        Performing augmentations is much faster on GPU than CPU
        '''

        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise 
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise 
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data 
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
            
        
        return features, n_time_steps

    def train(self):
        '''
        Train the model 
        '''

        # Set model to train mode (specificially to make sure dropout layers are engaged)
        self.model.train()

        # create vars to track performance
        train_losses = []
        val_losses = []
        val_DERs = []  # Track Diphone Error Rate
        val_PERs = []  # Track Phoneme Error Rate
        val_results = []

        val_steps_since_improvement = 0

        # training params 
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)

        early_stopping_val_steps = self.args['early_stopping_val_steps']

        train_start_time = time.time()

        # train for specified number of batches
        for i, batch in enumerate(self.train_loader):
            
            self.model.train()
            self.optimizer.zero_grad()
            
            # Train step
            start_time = time.time() 

            # Move data to device
            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            labels_phoneme = batch['seq_class_ids_phoneme'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            phone_seq_lens_phoneme = batch['phone_seq_lens_phoneme'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Use autocast for efficiency
            with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = self.amp_dtype):

                # Apply augmentations to the data
                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')

                adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model']['patch_stride'] + 1).to(torch.int32)

                # Get phoneme predictions 
                logits = self.model(features, day_indicies)

                # Check for extreme logits that could cause numerical issues
                if i % 200 == 0 or torch.any(torch.abs(logits) > 50):
                    logit_max = torch.max(torch.abs(logits)).item()
                    if logit_max > 50:
                        self.logger.warning(f"Batch {i}: Extreme logits detected! Max abs value: {logit_max:.2f}")

                # Calculate CTC Loss (Composite: diphone + phoneme)
                # Initialize for logging (will be overwritten if composite loss is used)
                diphone_loss = None
                phoneme_loss = None
                alpha = None  # Will be set if composite loss is used
                
                if self.args.get('use_composite_loss', False):
                    # Get current alpha (may be dynamic based on schedule)
                    alpha = self.get_composite_loss_alpha(i)
                    
                    # === DIPHONE LOSS ===
                    diphone_log_probs = logits.log_softmax(2)
                    diphone_loss = self.ctc_loss(
                        log_probs = torch.permute(diphone_log_probs, [1, 0, 2]),
                        targets = labels,
                        input_lengths = adjusted_lens,
                        target_lengths = phone_seq_lens
                    )
                    diphone_loss = torch.mean(diphone_loss)
                    
                    # === PHONEME LOSS ===
                    # Marginalize diphone logits to phoneme logits
                    phoneme_logits = self.marginalize_diphone_logits_to_phoneme_logits(logits)
                    phoneme_log_probs = phoneme_logits.log_softmax(2)
                    phoneme_loss = self.ctc_loss(
                        log_probs = torch.permute(phoneme_log_probs, [1, 0, 2]),
                        targets = labels_phoneme,
                        input_lengths = adjusted_lens,
                        target_lengths = phone_seq_lens_phoneme
                    )
                    phoneme_loss = torch.mean(phoneme_loss)
                    
                    # === COMPOSITE LOSS ===
                    loss = alpha * phoneme_loss + (1 - alpha) * diphone_loss
                    
                else:
                    # Original: Only diphone loss
                    log_probs = logits.log_softmax(2)
                    loss = self.ctc_loss(
                        log_probs = torch.permute(log_probs, [1, 0, 2]),
                        targets = labels,
                        input_lengths = adjusted_lens,
                        target_lengths = phone_seq_lens
                    )
                    loss = torch.mean(loss)
            
            # Check for NaN loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(f"NaN/Inf loss detected at batch {i}! Loss: {loss.item()}")
                self.logger.error(f"  Logit range: [{torch.min(logits).item():.2f}, {torch.max(logits).item():.2f}]")
                self.logger.error(f"  Input lengths: {adjusted_lens.cpu().tolist()}")
                self.logger.error(f"  Target lengths: {phone_seq_lens.cpu().tolist()}")
                self.logger.error("Skipping this batch and continuing training...")
                self.optimizer.zero_grad()
                continue
            
            loss.backward()

            # Clip gradient with robust handling of non-finite gradients
            if self.args['grad_norm_clip_value'] > 0: 
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               max_norm = self.args['grad_norm_clip_value'],
                                               error_if_nonfinite = False,  # Don't raise error, just clip
                                               foreach = True
                                               )
                
                # Log if gradients are non-finite
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.warning(f"Non-finite gradients detected at batch {i}! Grad norm: {grad_norm}")
                    self.logger.warning("Gradients have been clipped. Consider reducing learning rate.")

            self.optimizer.step()
            self.learning_rate_scheduler.step()
            
            # Save training metrics 
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            # Incrementally log training progress
            if i % self.args['batches_per_train_log'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_msg = (f'Train batch {i}: ' +
                          f'loss: {(loss.detach().item()):.2f} ')
                
                # If using composite loss, also log individual components and alpha
                if self.args.get('use_composite_loss', False):
                    log_msg += (f'(diphone: {diphone_loss.detach().item():.2f}, ' +
                               f'phoneme: {phoneme_loss.detach().item():.2f}, ' +
                               f'α: {alpha:.2f}) ')
                
                log_msg += (f'grad norm (pre-clip): {grad_norm:.2f} ' +
                           f'lr: {current_lr:.6f} ' +
                           f'time: {train_step_duration:.3f}')
                
                self.logger.info(log_msg)

            # Incrementally run a test step
            if i % self.args['batches_per_val_step'] == 0 or i == ((self.args['num_training_batches'] - 1)):
                self.logger.info(f"Running test after training batch: {i}")
                
                # Calculate metrics on val data
                start_time = time.time()
                val_metrics = self.validation(loader = self.val_loader, return_logits = self.args['save_val_logits'], return_data = self.args['save_val_data'])
                val_step_duration = time.time() - start_time


                # Log info 
                self.logger.info(f'Val batch {i}: ' +
                        f'DER (avg): {val_metrics["avg_DER"]:.4f} ' +
                        f'PER (avg): {val_metrics["avg_PER"]:.4f} ' +
                        f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} ' +
                        f'time: {val_step_duration:.3f}')
                
                if self.args['log_individual_day_val_PER']:
                    for day in val_metrics['day_DERs'].keys():
                        der = val_metrics['day_DERs'][day]['total_edit_distance'] / val_metrics['day_DERs'][day]['total_seq_length']
                        per = val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']
                        self.logger.info(f"{self.args['dataset']['sessions'][day]} - DER: {der:0.4f}, PER: {per:0.4f}")

                # Save metrics 
                val_DERs.append(val_metrics['avg_DER'])
                val_PERs.append(val_metrics['avg_PER'])
                val_losses.append(val_metrics['avg_loss'])
                val_results.append(val_metrics)

                # Determine if new best day. Based on if PER is lower, or in the case of a PER tie, if loss is lower
                new_best = False
                if val_metrics['avg_PER'] < self.best_val_PER:
                    self.logger.info(f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                    self.best_val_PER = val_metrics['avg_PER']
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True
                elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss): 
                    self.logger.info(f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True

                if new_best:

                    # Checkpoint if metrics have improved 
                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/best_checkpoint', self.best_val_PER, self.best_val_loss)

                    # save validation metrics to pickle file
                    if self.args['save_val_metrics']:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                            pickle.dump(val_metrics, f) 

                    val_steps_since_improvement = 0
                    
                else:
                    val_steps_since_improvement +=1

                # Optionally save this validation checkpoint, regardless of performance
                if self.args['save_all_val_steps']:
                    self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}', val_metrics['avg_PER'])

                # Early stopping 
                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(f'Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}')
                    break
                
        # Log final training steps 
        training_duration = time.time() - train_start_time


        self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        # Save final model 
        if self.args['save_final_model']:
            self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1])

        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses
        train_stats['val_DERs'] = val_DERs  # Diphone Error Rates
        train_stats['val_PERs'] = val_PERs  # Phoneme Error Rates
        train_stats['val_metrics'] = val_results

        return train_stats

    def validation(self, loader, return_logits = False, return_data = False):
        '''
        Calculate metrics on the validation dataset
        '''
        self.model.eval()

        metrics = {}
        
        # Record metrics
        if return_logits: 
            metrics['logits'] = []
            metrics['n_time_steps'] = []

        if return_data: 
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []  # Diphone sequences
        metrics['decoded_phoneme_seqs'] = []  # Phoneme sequences (NEW!)
        metrics['true_seq'] = []  # Ground truth diphone sequences
        metrics['true_phoneme_seqs'] = []  # Ground truth phoneme sequences (NEW!)
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []

        # Track both diphone and phoneme errors (with separate sequence lengths)
        total_diphone_edit_distance = 0
        total_phoneme_edit_distance = 0
        total_diphone_seq_length = 0
        total_phoneme_seq_length = 0

        # Calculate DER and PER for each specific day
        day_der = {}  # Diphone Error Rate per day
        day_per = {}  # Phoneme Error Rate per day
        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1: 
                day_der[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}
                day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for i, batch in enumerate(loader):        

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            labels_phoneme = batch['seq_class_ids_phoneme'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            phone_seq_lens_phoneme = batch['phone_seq_lens_phoneme'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Determine if we should perform validation on this batch
            day = day_indicies[0].item()
            if self.args['dataset']['dataset_probability_val'][day] == 0: 
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day}")
                continue
            
            with torch.no_grad():

                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = self.amp_dtype):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model']['patch_stride'] + 1).to(torch.int32)

                    logits = self.model(features, day_indicies)
    
                    loss = self.ctc_loss(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        labels,
                        adjusted_lens,
                        phone_seq_lens,
                    )
                    loss = torch.mean(loss)

                metrics['losses'].append(loss.cpu().detach().numpy())

                # Calculate DER (Diphone Error Rate) and PER (Phoneme Error Rate)
                batch_diphone_edit_distance = 0
                batch_phoneme_edit_distance = 0
                decoded_diphone_seqs = []
                decoded_phoneme_seqs = []
                true_phoneme_seqs = []
                
                for iterIdx in range(logits.shape[0]):
                    # === DIPHONE PREDICTIONS (DER) ===
                    # Argmax over diphone classes
                    decoded_diphone_seq = torch.argmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :].clone().detach(), dim=-1)
                    decoded_diphone_seq = torch.unique_consecutive(decoded_diphone_seq, dim=-1)
                    decoded_diphone_seq = decoded_diphone_seq.cpu().detach().numpy()
                    decoded_diphone_seq = np.array([i for i in decoded_diphone_seq if i != 0])
                    
                    # === PHONEME PREDICTIONS (PER) ===
                    # Convert diphone logits → probs → phoneme probs → phoneme predictions
                    diphone_probs = torch.softmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :], dim=-1)  # (time, 1681)
                    diphone_probs_np = diphone_probs.cpu().detach().numpy()
                    
                    # Marginalize to phoneme probabilities
                    # Add batch dimension for marginalize function: (1, time, 1681)
                    diphone_probs_3d = diphone_probs_np[np.newaxis, :, :]
                    phoneme_probs = marginalize_diphone_probabilities(diphone_probs_3d)  # (1, time, 41)
                    phoneme_probs = phoneme_probs[0]  # Remove batch dim: (time, 41)
                    
                    # Argmax to get most likely phoneme at each timestep
                    decoded_phoneme_seq = np.argmax(phoneme_probs, axis=-1)  # (time,)
                    
                    # CTC collapse: remove consecutive duplicates
                    decoded_phoneme_seq = np.array([decoded_phoneme_seq[0]] + 
                                                   [decoded_phoneme_seq[i] for i in range(1, len(decoded_phoneme_seq)) 
                                                    if decoded_phoneme_seq[i] != decoded_phoneme_seq[i-1]])
                    # Remove blanks (0)
                    decoded_phoneme_seq = np.array([i for i in decoded_phoneme_seq if i != 0])

                    # === GROUND TRUTH ===
                    # Get ground truth diphone sequence
                    true_diphone_seq = np.array(
                        labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                    )
                    
                    # Get ground truth phoneme sequence directly from data
                    true_phoneme_seq = np.array(
                        labels_phoneme[iterIdx][0 : phone_seq_lens_phoneme[iterIdx]].cpu().detach()
                    )
            
                    # Compute edit distances
                    # DER: Compare predicted diphones vs ground truth diphones
                    batch_diphone_edit_distance += F.edit_distance(decoded_diphone_seq, true_diphone_seq)
                    
                    # PER: Compare predicted phonemes vs ground truth phonemes
                    batch_phoneme_edit_distance += F.edit_distance(decoded_phoneme_seq, true_phoneme_seq)

                    decoded_diphone_seqs.append(decoded_diphone_seq)
                    decoded_phoneme_seqs.append(decoded_phoneme_seq)
                    true_phoneme_seqs.append(true_phoneme_seq)

            day = batch['day_indicies'][0].item()
            
            # Calculate actual sequence lengths for this batch
            batch_diphone_seq_length = torch.sum(phone_seq_lens).item()  # Diphone sequence length
            batch_phoneme_seq_length = sum(len(seq) for seq in true_phoneme_seqs)  # Phoneme sequence length
                
            # Accumulate errors for this day
            day_der[day]['total_edit_distance'] += batch_diphone_edit_distance
            day_der[day]['total_seq_length'] += batch_diphone_seq_length
            
            day_per[day]['total_edit_distance'] += batch_phoneme_edit_distance
            day_per[day]['total_seq_length'] += batch_phoneme_seq_length

            # Accumulate total errors
            total_diphone_edit_distance += batch_diphone_edit_distance
            total_phoneme_edit_distance += batch_phoneme_edit_distance
            total_diphone_seq_length += batch_diphone_seq_length
            total_phoneme_seq_length += batch_phoneme_seq_length

            # Record metrics
            if return_logits: 
                metrics['logits'].append(logits.cpu().float().numpy()) # Will be in bfloat16 if AMP is enabled, so need to set back to float32
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())

            if return_data: 
                metrics['input_features'].append(batch['input_features'].cpu().numpy()) 

            metrics['decoded_seqs'].append(decoded_diphone_seqs)  # Diphone predictions
            metrics['decoded_phoneme_seqs'].append(decoded_phoneme_seqs)  # Phoneme predictions
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())  # Ground truth diphones
            metrics['true_phoneme_seqs'].append(true_phoneme_seqs)  # Ground truth phonemes
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

        # Compute average error rates (using appropriate sequence lengths)
        avg_DER = total_diphone_edit_distance / total_diphone_seq_length
        avg_PER = total_phoneme_edit_distance / total_phoneme_seq_length

        # Store both metrics
        metrics['day_DERs'] = day_der  # Diphone Error Rate per day
        metrics['day_PERs'] = day_per  # Phoneme Error Rate per day
        metrics['avg_DER'] = avg_DER  # Average Diphone Error Rate
        metrics['avg_PER'] = avg_PER  # Average Phoneme Error Rate
        metrics['avg_loss'] = np.mean(metrics['losses'])

        return metrics