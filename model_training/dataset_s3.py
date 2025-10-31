"""
Modified dataset that reads HDF5 files directly from S3
"""
import os
import torch
from torch.utils.data import Dataset 
import h5py
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import math
import s3fs


class BrainToTextDatasetS3(Dataset):
    '''
    Dataset for brain-to-text data that reads directly from S3
    
    Returns an entire batch of data instead of a single example
    '''

    def __init__(
            self, 
            trial_indicies,
            n_batches,
            s3_filesystem=None,
            split='train', 
            batch_size=64, 
            days_per_batch=1, 
            random_seed=-1,
            must_include_days=None,
            feature_subset=None
            ): 
        '''
        trial_indicies:  (dict)      - dictionary with day numbers as keys and lists of trial indices as values
        n_batches:       (int)       - number of random training batches to create
        s3_filesystem:   (s3fs.S3FileSystem) - S3 filesystem object for reading from S3
        split:           (string)    - string specifying if this is a train or test dataset
        batch_size:      (int)       - number of examples to include in batch returned from __getitem_()
        days_per_batch:  (int)       - how many unique days can exist in a batch
        random_seed:     (int)       - seed to set for randomly assigning trials to a batch
        must_include_days ([int])    - list of days that must be included in every batch
        feature_subset  ([int])      - list of neural feature indicies that should be the only features included
        '''
        
        # Set random seed for reproducibility
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.split = split
        self.s3fs = s3_filesystem

        # Ensure the split is valid
        if self.split not in ['train', 'test']:
            raise ValueError(f'split must be either "train" or "test". Received {self.split}')
        
        self.days_per_batch = days_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.days = {}
        self.n_trials = 0 
        self.trial_indicies = trial_indicies
        self.n_days = len(trial_indicies.keys())
        self.feature_subset = feature_subset

        # Calculate total number of trials in the dataset
        for d in trial_indicies:
            self.n_trials += len(trial_indicies[d]['trials'])

        if must_include_days is not None and len(must_include_days) > days_per_batch:
            raise ValueError(f'must_include_days must be less than or equal to days_per_batch')
        
        if must_include_days is not None and len(must_include_days) > self.n_days and split != 'train':
            raise ValueError(f'must_include_days is not valid for test data')
        
        if must_include_days is not None:
            # Map must_include_days to correct indicies if they are negative
            for i, d in enumerate(must_include_days):
                if d < 0: 
                    must_include_days[i] = self.n_days + d

        self.must_include_days = must_include_days    

        # Ensure that the days_per_batch is not greater than the number of days in the dataset
        if self.split == 'train' and self.days_per_batch > self.n_days:
            raise ValueError(f'Requested days_per_batch: {days_per_batch} is greater than available days {self.n_days}.')
           
        if self.split == 'train':
            self.batch_index = self.create_batch_index_train()
        else: 
            self.batch_index = self.create_batch_index_test()
            self.n_batches = len(self.batch_index.keys())
    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        ''' 
        Gets an entire batch of data from the dataset, not just a single item
        Reads directly from S3 using s3fs
        '''
        batch = {
            'input_features': [],
            'seq_class_ids': [],
            'n_time_steps': [],
            'phone_seq_lens': [],
            'day_indicies': [],
            'transcriptions': [],
            'block_nums': [],
            'trial_nums': [],
        }

        index = self.batch_index[idx]

        # Iterate through each day in the index
        for d in index.keys():
            session_path = self.trial_indicies[d]['session_path']
            
            # Open the hdf5 file from S3
            try:
                if self.s3fs is not None:
                    # Read from S3 using s3fs
                    with self.s3fs.open(session_path, 'rb') as s3_file:
                        with h5py.File(s3_file, 'r') as f:
                            self._load_trials_from_file(f, index[d], d, batch)
                else:
                    # Read from local filesystem
                    with h5py.File(session_path, 'r') as f:
                        self._load_trials_from_file(f, index[d], d, batch)
                        
            except Exception as e:
                print(f'Error opening file {session_path}: {e}')
                continue

        # Pad data to form a cohesive batch
        if len(batch['input_features']) == 0:
            raise RuntimeError(f"No data loaded for batch {idx}")
            
        batch['input_features'] = pad_sequence(batch['input_features'], batch_first=True, padding_value=0)
        batch['seq_class_ids'] = pad_sequence(batch['seq_class_ids'], batch_first=True, padding_value=0)

        batch['n_time_steps'] = torch.tensor(batch['n_time_steps']) 
        batch['phone_seq_lens'] = torch.tensor(batch['phone_seq_lens'])
        batch['day_indicies'] = torch.tensor(batch['day_indicies'])
        batch['transcriptions'] = torch.stack(batch['transcriptions'])
        batch['block_nums'] = torch.tensor(batch['block_nums'])
        batch['trial_nums'] = torch.tensor(batch['trial_nums'])

        return batch
    
    def _load_trials_from_file(self, f, trial_list, day_idx, batch):
        """
        Helper method to load trials from an open HDF5 file
        """
        for t in trial_list:
            try: 
                g = f[f'trial_{t:04d}']

                # Remove features if necessary 
                input_features = torch.from_numpy(g['input_features'][:])
                if self.feature_subset:
                    input_features = input_features[:, self.feature_subset]

                batch['input_features'].append(input_features)
                batch['seq_class_ids'].append(torch.from_numpy(g['seq_class_ids'][:]))
                batch['transcriptions'].append(torch.from_numpy(g['transcription'][:]))
                batch['n_time_steps'].append(g.attrs['n_time_steps'])
                batch['phone_seq_lens'].append(g.attrs['seq_len'])
                batch['day_indicies'].append(int(day_idx))
                batch['block_nums'].append(g.attrs['block_num'])
                batch['trial_nums'].append(g.attrs['trial_num'])
            
            except Exception as e:
                print(f'Error loading trial {t}: {e}')
                continue

    def create_batch_index_train(self):
        '''
        Create an index that maps a batch_number to batch_size number of trials
        '''
        batch_index = {}

        # Precompute the days that are not in must_include_days
        if self.must_include_days is not None:
            non_must_include_days = [d for d in self.trial_indicies.keys() if d not in self.must_include_days]

        for batch_idx in range(self.n_batches):
            batch = {}

            if self.must_include_days is not None and len(self.must_include_days) > 0:
                days = np.concatenate((self.must_include_days, np.random.choice(non_must_include_days, size=self.days_per_batch - len(self.must_include_days), replace=False)))
            else: 
                days = np.random.choice(list(self.trial_indicies.keys()), size=self.days_per_batch, replace=False)
            
            num_trials = math.ceil(self.batch_size / self.days_per_batch)

            for d in days:
                trial_idxs = np.random.choice(self.trial_indicies[d]['trials'], size=num_trials, replace=True)
                batch[d] = trial_idxs

            # Remove extra trials
            extra_trials = (num_trials * len(days)) - self.batch_size

            while extra_trials > 0: 
                d = np.random.choice(days)
                batch[d] = batch[d][:-1]
                extra_trials -= 1

            batch_index[batch_idx] = batch

        return batch_index
    
    def create_batch_index_test(self):
        '''
        Create an index that is all validation/testing data in batches of up to self.batch_size
        '''
        batch_index = {}
        batch_idx = 0
        
        for d in self.trial_indicies.keys():
            num_trials = len(self.trial_indicies[d]['trials'])
            num_batches = (num_trials + self.batch_size - 1) // self.batch_size 
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_trials)
                
                batch_trials = self.trial_indicies[d]['trials'][start_idx:end_idx]
                batch_index[batch_idx] = {d: batch_trials}
                batch_idx += 1
        
        return batch_index


def train_test_split_indicies_s3(file_paths, s3_filesystem=None, test_percentage=0.1, seed=-1, bad_trials_dict=None):
    '''
    Split data from file_paths into train and test splits
    Works with both S3 and local file paths
    
    Args:
        file_paths (list): List of file paths (S3 or local) to the hdf5 files
        s3_filesystem (s3fs.S3FileSystem): S3 filesystem object if using S3
        test_percentage (float): Percentage of trials to use for testing
        seed (int): Seed for reproducibility
        bad_trials_dict (dict): Dictionary of trials to exclude
    '''
    if seed != -1:
        np.random.seed(seed)

    trials_per_day = {}
    for i, path in enumerate(file_paths):
        session = [s for s in path.split('/') if (s.startswith('t15.20') or s.startswith('t12.20'))][0]

        good_trial_indices = []

        try:
            if s3_filesystem is not None:
                # Read from S3
                with s3_filesystem.open(path, 'rb') as s3_file:
                    with h5py.File(s3_file, 'r') as f:
                        good_trial_indices = _extract_good_trials(f, session, bad_trials_dict)
            else:
                # Read from local filesystem
                if os.path.exists(path):
                    with h5py.File(path, 'r') as f:
                        good_trial_indices = _extract_good_trials(f, session, bad_trials_dict)
                        
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        trials_per_day[i] = {
            'num_trials': len(good_trial_indices), 
            'trial_indices': good_trial_indices, 
            'session_path': path
        }

    # Split into train and test
    train_trials = {}
    test_trials = {}

    for day in trials_per_day.keys():
        num_trials = trials_per_day[day]['num_trials']
        all_trial_indices = trials_per_day[day]['trial_indices']

        if test_percentage == 0:
            train_trials[day] = {'trials': all_trial_indices, 'session_path': trials_per_day[day]['session_path']}
            test_trials[day] = {'trials': [], 'session_path': trials_per_day[day]['session_path']}
        elif test_percentage == 1:
            train_trials[day] = {'trials': [], 'session_path': trials_per_day[day]['session_path']}
            test_trials[day] = {'trials': all_trial_indices, 'session_path': trials_per_day[day]['session_path']}
        else:
            num_test = max(1, int(num_trials * test_percentage))
            test_indices = np.random.choice(all_trial_indices, size=num_test, replace=False).tolist()
            train_indices = [idx for idx in all_trial_indices if idx not in test_indices]
            
            train_trials[day] = {'trials': train_indices, 'session_path': trials_per_day[day]['session_path']}
            test_trials[day] = {'trials': test_indices, 'session_path': trials_per_day[day]['session_path']}
    
    return train_trials, test_trials


def _extract_good_trials(f, session, bad_trials_dict):
    """
    Helper function to extract good trial indices from an HDF5 file
    """
    good_trial_indices = []
    num_trials = len(list(f.keys()))
    
    for t in range(num_trials):
        key = f'trial_{t:04d}'
        
        if key not in f:
            continue
            
        block_num = f[key].attrs['block_num']
        trial_num = f[key].attrs['trial_num']

        if (
            bad_trials_dict is not None
            and session in bad_trials_dict
            and str(block_num) in bad_trials_dict[session]
            and trial_num in bad_trials_dict[session][str(block_num)]
        ):
            continue

        good_trial_indices.append(t)
    
    return good_trial_indices

