# Jupyter configuration to disable checkpoint creation in model_training directory
# This prevents .ipynb_checkpoints from being created

c = get_config()

# Disable saving checkpoints
c.FileContentsManager.checkpoints_kwargs = {'root_dir': '/tmp'}

# Or completely disable checkpoints
# c.FileCheckpoints.checkpoint_dir = '/tmp/.checkpoints'

