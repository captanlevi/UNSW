import numpy as np
import pickle

def getLabelMapping(labels : list):
    """
    gives label mapping
    """
    unique_labels = np.unique(labels)
    label_mapping = dict()
    i = 0
    for unique_label in unique_labels:
        label_mapping[unique_label] = i
        i += 1
    return label_mapping


def saveLVectors(path,l_vectors):
    with open(path, "wb") as f:
        pickle.dump(l_vectors,f)
        
def loadLVectors(path):
    with open(path, "rb") as f:
        l_vectors = pickle.load(f)
    return l_vectors