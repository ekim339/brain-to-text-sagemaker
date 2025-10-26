from omegaconf import OmegaConf
from rnn_trainer_s3 import BrainToTextDecoder_Trainer_S3

args = OmegaConf.load('rnn_args_diphone_sagemaker.yaml')
trainer = BrainToTextDecoder_Trainer_S3(args)
metrics = trainer.train()