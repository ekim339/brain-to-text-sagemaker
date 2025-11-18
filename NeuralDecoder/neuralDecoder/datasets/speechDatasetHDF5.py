import pathlib
import random
import numpy as np
import tensorflow as tf
import h5py

PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

PHONE_DEF_SIL = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', 'SIL'
]

class SpeechDatasetHDF5():
    """Dataset loader for HDF5 encoded speech data."""
    
    def __init__(self,
                 rawFileDir,
                 nInputFeatures,
                 nClasses,
                 maxSeqElements,
                 bufferSize,
                 syntheticFileDir=None,
                 syntheticMixingRate=0.33,
                 subsetSize=-1,
                 labelDir=None,
                 timeWarpSmoothSD=0.0,
                 timeWarpNoiseSD=0.0,
                 chanIndices=None
                 ):

        self.rawFileDir = rawFileDir
        self.nInputFeatures = nInputFeatures
        self.nClasses = nClasses
        self.maxSeqElements = maxSeqElements
        self.bufferSize = bufferSize
        self.syntheticFileDir = syntheticFileDir
        self.syntheticMixingRate = syntheticMixingRate
        self.timeWarpSmoothSD = timeWarpSmoothSD
        self.timeWarpNoiseSD = timeWarpNoiseSD
        self.subsetSize = subsetSize
        self.chanIndices = chanIndices
        
    def _load_hdf5_data(self, file_path):
        """Load data from an HDF5 file."""
        try:
            with h5py.File(file_path, 'r') as f:
                # Print available keys for debugging
                print(f"HDF5 file keys: {list(f.keys())}")
                
                # Extract all keys/datasets from HDF5
                data_dict = {}
                for key in f.keys():
                    try:
                        data_dict[key] = f[key][:]
                        print(f"  Loaded '{key}': shape={data_dict[key].shape}, dtype={data_dict[key].dtype}")
                    except Exception as e:
                        print(f"  Warning: Could not load key '{key}': {e}")
                
                return data_dict
        except Exception as e:
            raise RuntimeError(f"Failed to load HDF5 file {file_path}: {e}")
        
    def build(self, batchSize, isTraining):
        """Build the dataset pipeline."""
        
        # The rawFileDir comes in as: /path/to/hdf5_data_encoded/session_name/train
        # But HDF5 files are at: /path/to/hdf5_data_encoded/session_name/data_train.hdf5
        # So we need to go up one directory from 'train' to get to the session directory
        
        raw_path = pathlib.Path(self.rawFileDir)
        
        # If the path ends with 'train' or 'test', go up one directory
        if raw_path.name in ['train', 'test', 'competitionHoldOut']:
            session_dir = raw_path.parent
        else:
            session_dir = raw_path
        
        # Determine which HDF5 file to load based on training mode
        if isTraining:
            hdf5_filename = 'data_train.hdf5'
        else:
            hdf5_filename = 'data_test.hdf5'
            
        hdf5_path = session_dir / hdf5_filename
        
        print(f'Loading HDF5 data from {hdf5_path}')
        
        if not hdf5_path.exists():
            # Try alternative file names
            alt_files = list(session_dir.glob('*.hdf5'))
            if alt_files:
                hdf5_path = alt_files[0]
                print(f'File not found, using alternative: {hdf5_path}')
            else:
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}\nSearched in: {session_dir}")
        
        # Load HDF5 data
        hdf5_data = self._load_hdf5_data(str(hdf5_path))
        
        # Create a generator function to yield individual examples
        def data_generator():
            n_examples = len(hdf5_data.get('inputFeatures', []))
            indices = list(range(n_examples))
            
            if isTraining:
                random.shuffle(indices)
            
            for idx in indices:
                example = {}
                for key in hdf5_data.keys():
                    if isinstance(hdf5_data[key], (list, np.ndarray)):
                        if len(hdf5_data[key]) > idx:
                            example[key] = hdf5_data[key][idx]
                
                yield example
        
        # Infer output signature from the HDF5 data
        output_signature = {}
        for key in hdf5_data.keys():
            if isinstance(hdf5_data[key], (list, np.ndarray)) and len(hdf5_data[key]) > 0:
                sample = hdf5_data[key][0]
                if isinstance(sample, np.ndarray):
                    output_signature[key] = tf.TensorSpec(shape=(None,) + sample.shape, dtype=tf.float32)
                else:
                    output_signature[key] = tf.TensorSpec(shape=(None,), dtype=tf.int64)
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        )
        
        # Apply channel selection if specified
        if self.chanIndices is not None:
            def select_channels(example):
                if 'inputFeatures' in example:
                    selected = tf.gather(example['inputFeatures'], tf.constant(self.chanIndices), axis=-1)
                    paddings = [[0, 0], [0, self.nInputFeatures - tf.shape(selected)[-1]]]
                    example['inputFeatures'] = tf.pad(selected, paddings, 'CONSTANT', constant_values=0)
                return example
            
            dataset = dataset.map(select_channels, num_parallel_calls=tf.data.AUTOTUNE)
        
        if isTraining:
            # For training: create adaptation dataset
            datasetForAdapt = dataset.map(
                lambda x: x.get('inputFeatures', x) + 0.001,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Take subset if specified
            if self.subsetSize > 0:
                dataset = dataset.take(self.subsetSize)
            
            # Shuffle and repeat
            dataset = dataset.shuffle(self.bufferSize)
            dataset = dataset.repeat()
            dataset = dataset.padded_batch(batchSize)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset, datasetForAdapt
        else:
            # For evaluation: just batch
            dataset = dataset.padded_batch(batchSize)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset

