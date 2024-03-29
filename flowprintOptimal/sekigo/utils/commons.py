import numpy as np

def downSampleArray(array : np.ndarray,factor):
    """
    Array shape is (......,time_steps) needs to be atleast 2D
    Changes array inplace
    """
    assert array.shape[1] >= factor
    length = array.shape[1] - array.shape[1]%factor
    down_sampled_array = array[:,:length].reshape(array.shape[0],-1,factor).sum(axis = -1)
    return down_sampled_array
