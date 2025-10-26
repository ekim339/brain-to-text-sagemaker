"""
Modified trainer that works with S3-based datasets
Inherits from the base trainer and overrides dataset initialization
"""
import torch 
from torch.utils.data import DataLoader
import os
import json
import s3fs

from rnn_trainer import BrainToTextDecoder_Trainer
from dataset_s3 import BrainToTextDatasetS3, train_test_split_indicies_s3


class BrainToTextDecoder_Trainer_S3(BrainToTextDecoder_Trainer):
    """
    Extension of BrainToTextDecoder_Trainer that reads data directly from S3
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        Must include:
            - s3_bucket: S3 bucket name
            - s3_data_prefix: S3 prefix for data (e.g., 'data/hdf5_data_diphone_encoded')
        '''
        
        # Initialize S3 filesystem
        self.s3_bucket = args.get('s3_bucket', None)
        self.s3_data_prefix = args.get('s3_data_prefix', None)
        
        if self.s3_bucket and self.s3_data_prefix:
            print(f"Initializing S3 filesystem for bucket: {self.s3_bucket}")
            self.s3fs = s3fs.S3FileSystem(anon=False)  # Uses AWS credentials from environment
            print("S3 filesystem initialized successfully")
        else:
            self.s3fs = None
            print("No S3 configuration provided, using local filesystem")
        
        # Store original dataset_dir
        self.original_dataset_dir = args['dataset']['dataset_dir']
        
        # Call parent __init__ but we'll override the dataset creation
        # First, temporarily set a dummy dataset_dir to avoid errors
        args['dataset']['dataset_dir'] = '/tmp/dummy'
        
        # We need to initialize everything EXCEPT datasets
        # So we'll call the parent init but catch the dataset creation
        self._init_without_datasets(args)
        
        # Now create S3-aware datasets
        self._create_s3_datasets()
    
    def _init_without_datasets(self, args):
        """
        Initialize everything from parent except datasets
        """
        # Copy the entire parent __init__ logic but skip dataset creation
        self.args = args
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 

        self.best_val_PER = torch.inf
        self.best_val_loss = torch.inf

        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 

        self.transform_args = self.args['dataset']['data_transforms']

        # Import everything we need
        import logging
        import sys
        import pathlib
        import random
        import numpy as np
        from rnn_model import GRUDecoder
        
        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=True)

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']: 
            os.makedirs(self.args['checkpoint_dir'], exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device
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

        # Set seed
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        # Initialize the model 
        self.model = GRUDecoder(
            neural_dim=self.args['model']['n_input_features'],
            n_units=self.args['model']['n_units'],
            n_days=len(self.args['dataset']['sessions']),
            n_classes=self.args['dataset']['n_classes'],
            rnn_dropout=self.args['model']['rnn_dropout'], 
            input_dropout=self.args['model']['input_network']['input_layer_dropout'], 
            n_layers=self.args['model']['n_layers'],
            patch_size=self.args['model']['patch_size'],
            patch_stride=self.args['model']['patch_stride'],
        )

        self.logger.info("Using torch.compile")
        self.model = torch.compile(self.model)
        self.logger.info(f"Initialized RNN decoding model")
        self.logger.info(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        day_params = sum(p.numel() for name, p in self.model.named_parameters() if 'day' in name)
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total")

        # Note: Datasets will be created in _create_s3_datasets()
        
        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            self.learning_rate_scheduler = LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args['lr_min'] / self.args['lr_max'],
                total_iters=self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")
        
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=False)

        # Load from checkpoint if specified
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Freeze layers if specified
        for name, param in self.model.named_parameters():
            if not self.args['model']['rnn_trainable'] and 'gru' in name:
                param.requires_grad = False
            elif not self.args['model']['input_network']['input_trainable'] and 'day' in name:
                param.requires_grad = False

        # Send model to device 
        self.model.to(self.device)

    def _create_s3_datasets(self):
        """
        Create datasets that read from S3
        """
        # Build S3 paths
        if self.s3fs is not None:
            base_path = f"{self.s3_bucket}/{self.s3_data_prefix}"
        else:
            base_path = self.original_dataset_dir
            
        train_file_paths = [
            f"{base_path}/{s}/data_train.hdf5" 
            for s in self.args['dataset']['sessions']
        ]
        val_file_paths = [
            f"{base_path}/{s}/data_val.hdf5" 
            for s in self.args['dataset']['sessions']
        ]

        self.logger.info(f"Loading data from: {base_path}")
        self.logger.info(f"Example train path: {train_file_paths[0]}")

        # Ensure no duplicates
        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions in train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions in val dataset")

        # Split trials into train and test sets
        self.logger.info("Creating train/val split indices...")
        train_trials, _ = train_test_split_indicies_s3(
            file_paths=train_file_paths, 
            s3_filesystem=self.s3fs,
            test_percentage=0,
            seed=self.args['dataset']['seed'],
            bad_trials_dict=None,
        )
        _, val_trials = train_test_split_indicies_s3(
            file_paths=val_file_paths, 
            s3_filesystem=self.s3fs,
            test_percentage=1,
            seed=self.args['dataset']['seed'],
            bad_trials_dict=None,
        )

        # Save trial split info
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            # Convert to serializable format (remove s3fs references)
            serializable_train = {k: {'trials': v['trials'], 'session_path': v['session_path']} for k, v in train_trials.items()}
            serializable_val = {k: {'trials': v['trials'], 'session_path': v['session_path']} for k, v in val_trials.items()}
            json.dump({'train': serializable_train, 'val': serializable_val}, f)

        # Feature subset
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] is not None: 
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using feature subset: {feature_subset}')
            
        # Create train dataset and dataloader
        self.logger.info("Creating training dataset...")
        self.train_dataset = BrainToTextDatasetS3(
            trial_indicies=train_trials,
            split='train',
            s3_filesystem=self.s3fs,
            days_per_batch=self.args['dataset']['days_per_batch'],
            n_batches=self.args['num_training_batches'],
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=self.args['dataset']['loader_shuffle'],
            num_workers=self.args['dataset']['num_dataloader_workers'],
            pin_memory=True 
        )

        # Create val dataset and dataloader
        self.logger.info("Creating validation dataset...")
        self.val_dataset = BrainToTextDatasetS3(
            trial_indicies=val_trials, 
            split='test',
            s3_filesystem=self.s3fs,
            days_per_batch=None,
            n_batches=None,
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset   
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False, 
            num_workers=0,
            pin_memory=True 
        )

        self.logger.info("Successfully initialized S3-backed datasets")
        self.logger.info(f"Train batches: {len(self.train_dataset)}")
        self.logger.info(f"Val batches: {len(self.val_dataset)}")

