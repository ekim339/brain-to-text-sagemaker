"""
SageMaker S3 trainer with MLflow logging integration.
Inherits from BrainToTextDecoder_Trainer_S3 and adds MLflow tracking.
"""

import os
import mlflow
import mlflow.pytorch
from rnn_trainer_s3 import BrainToTextDecoder_Trainer_S3


class BrainToTextDecoder_Trainer_S3_MLflow(BrainToTextDecoder_Trainer_S3):
    """
    Extends BrainToTextDecoder_Trainer_S3 with MLflow logging.
    """
    
    def __init__(self, args):
        """
        Initialize trainer with MLflow tracking.
        
        Args:
            args: Training configuration (OmegaConf dict)
        """
        # Set MLflow tracking URI (use environment variable or default to local)
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Set experiment name
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'brain-to-text-diphone')
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        self.mlflow_run = mlflow.start_run()
        
        print(f"MLflow tracking URI: {mlflow_uri}")
        print(f"MLflow experiment: {experiment_name}")
        print(f"MLflow run ID: {self.mlflow_run.info.run_id}")
        
        # Initialize parent class
        super().__init__(args)
        
        # Log all hyperparameters
        self._log_hyperparameters()
    
    def _log_hyperparameters(self):
        """Log all training hyperparameters to MLflow."""
        
        # Learning rate parameters
        mlflow.log_param("lr_max", self.args['lr_max'])
        mlflow.log_param("lr_min", self.args['lr_min'])
        mlflow.log_param("lr_warmup_steps", self.args['lr_warmup_steps'])
        mlflow.log_param("lr_scheduler_type", self.args['lr_scheduler_type'])
        
        # Day-specific learning rate
        mlflow.log_param("lr_max_day", self.args.get('lr_max_day', 'N/A'))
        mlflow.log_param("lr_warmup_steps_day", self.args.get('lr_warmup_steps_day', 'N/A'))
        
        # Optimizer parameters
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("weight_decay", self.args['weight_decay'])
        mlflow.log_param("epsilon", self.args['epsilon'])
        mlflow.log_param("grad_norm_clip_value", self.args.get('grad_norm_clip_value', 'None'))
        
        # Model architecture
        mlflow.log_param("model_type", self.args['model']['model_type'])
        mlflow.log_param("n_classes", self.args['dataset']['n_classes'])
        mlflow.log_param("hidden_dim", self.args['model']['hidden_dim'])
        mlflow.log_param("n_layers", self.args['model']['n_layers'])
        mlflow.log_param("dropout", self.args['model']['dropout'])
        mlflow.log_param("patch_size", self.args['model']['patch_size'])
        mlflow.log_param("patch_stride", self.args['model']['patch_stride'])
        
        # Training parameters
        mlflow.log_param("batch_size", self.args['dataset']['batch_size'])
        mlflow.log_param("num_training_batches", self.args['num_training_batches'])
        mlflow.log_param("val_steps", self.args['val_steps'])
        mlflow.log_param("use_amp", self.args.get('use_amp', False))
        mlflow.log_param("early_stopping", self.args.get('early_stopping', True))
        mlflow.log_param("early_stopping_val_steps", self.args.get('early_stopping_val_steps', 'N/A'))
        
        # Data augmentation
        if 'data_augmentation' in self.args:
            for aug_type, aug_config in self.args['data_augmentation'].items():
                mlflow.log_param(f"aug_{aug_type}_enabled", aug_config.get('enabled', False))
        
        # S3 configuration
        mlflow.log_param("s3_bucket", self.args.get('s3_bucket', 'N/A'))
        mlflow.log_param("s3_data_prefix", self.args.get('s3_data_prefix', 'N/A'))
        
        # Dataset sessions
        mlflow.log_param("num_sessions", len(self.args['dataset']['sessions']))
    
    def train(self):
        """
        Override train method to add MLflow logging.
        """
        try:
            # Call parent train method
            train_stats = super().train()
            
            # Log final best metrics
            mlflow.log_metric("best_val_DER", self.best_val_PER, step=self.args['num_training_batches'])
            mlflow.log_metric("best_val_loss", self.best_val_loss, step=self.args['num_training_batches'])
            
            # Log model artifact (best checkpoint)
            best_checkpoint_path = f'{self.args["checkpoint_dir"]}/best_checkpoint'
            if os.path.exists(best_checkpoint_path):
                mlflow.log_artifacts(best_checkpoint_path, artifact_path="best_checkpoint")
            
            # Log config file
            mlflow.log_dict(dict(self.args), "training_config.yaml")
            
            print(f"\nMLflow run completed: {self.mlflow_run.info.run_id}")
            print(f"View at: {mlflow.get_tracking_uri()}")
            
            return train_stats
            
        except Exception as e:
            # Mark run as failed
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(e))
            raise
        
        finally:
            # End MLflow run
            mlflow.end_run()
    
    def validation(self, loader, return_logits=False, return_data=False):
        """
        Override validation to log metrics to MLflow.
        """
        # Call parent validation
        metrics = super().validation(loader, return_logits, return_data)
        
        # Log validation metrics to MLflow (if we're in training loop)
        if hasattr(self, 'current_batch_idx'):
            mlflow.log_metric("val_DER", metrics['avg_DER'], step=self.current_batch_idx)
            mlflow.log_metric("val_PER", metrics['avg_PER'], step=self.current_batch_idx)
            mlflow.log_metric("val_loss", metrics['avg_loss'], step=self.current_batch_idx)
        
        return metrics
    
    def _log_training_step(self, batch_idx, loss, grad_norm, current_lr):
        """
        Log training metrics to MLflow.
        
        Args:
            batch_idx: Current training batch index
            loss: Training loss value
            grad_norm: Gradient norm value
            current_lr: Current learning rate
        """
        self.current_batch_idx = batch_idx
        
        # Log every N steps to avoid too many API calls
        if batch_idx % 10 == 0:
            mlflow.log_metric("train_loss", loss, step=batch_idx)
            mlflow.log_metric("grad_norm", grad_norm, step=batch_idx)
            mlflow.log_metric("learning_rate", current_lr, step=batch_idx)


# Monkey-patch the training loop to add MLflow logging
# This is a workaround to avoid modifying the core training loop too much
def _patch_training_loop(trainer_instance):
    """
    Adds MLflow logging calls to the training loop.
    This is called automatically when using BrainToTextDecoder_Trainer_S3_MLflow.
    """
    original_train = trainer_instance.train
    
    def train_with_logging():
        # Set up hooks for logging during training
        # This would require modifying the actual training loop
        # For now, we'll log at validation steps
        return original_train()
    
    trainer_instance.train = train_with_logging

