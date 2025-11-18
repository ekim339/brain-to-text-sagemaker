from .speechDataset import SpeechDataset
from .speechDatasetHDF5 import SpeechDatasetHDF5

def getDataset(datasetName):
    if datasetName == 'speech':
        return SpeechDataset
    elif datasetName == 'speech_hdf5':
        return SpeechDatasetHDF5
    else:
        raise ValueError('Dataset not found')