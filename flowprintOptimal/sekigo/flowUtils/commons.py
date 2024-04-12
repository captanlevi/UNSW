from ..core.flowRepresentation import FlowRepresentation
from typing import List
import json
import numpy as np

def saveFlows(path,flows : List[FlowRepresentation]):
    with open(path,"w") as f:
        serilized_data = list(map(lambda x : json.dumps(x.ser()),flows))
        json.dump(serilized_data,f)

def loadFlows(path) -> List[FlowRepresentation]:
    with open(path,"r") as f:
        data = json.load(f)
        flows = list(map(lambda x : FlowRepresentation.deSer(json.loads(x)), data))
    return flows



def prefixConvolution(array, window_size):
    """Perform a convolution where the result is aligned with the first element of the window."""
    # Manually pad the end of the array to ensure the convolution result aligns as desired
    window = np.ones(int(window_size))
    pad_width = len(window) - 1
    padded_array = np.pad(array, (0, pad_width), mode='constant', constant_values=0)
    # Perform the convolution
    result = np.convolve(padded_array, window, 'valid')
    return result



def getValidInvalidStartingPointsForSubFlowStart(flow : FlowRepresentation,required_length,min_activity : int):
    """
    Valid points must have atleast some activity hapenning in the required length region.
    Activity is the number of packets in either direction
    """
    sum_array = (flow.up_packets +  flow.down_packets).sum(axis = 0)
    data_point_length = sum_array.shape[0]
    max_start_index = data_point_length - required_length
    convoluted = prefixConvolution(sum_array,required_length)
    valid_points = []
    invalid_points = []
    for i in range(max_start_index):
        if convoluted[i] >= min_activity:
            valid_points.append(i)
        else:
            invalid_points.append(i)
    
    return valid_points,invalid_points





def getActivityArrayFromFlow(flow : FlowRepresentation):
    """
    Takes a flow and returns a global normalized array
    Dividing the packetlength by the band_thresholds 

    returns array of shape ((num_thresholds)*2, seq_lens)
    """
    band_thresholds = flow.flow_config.band_thresholds[:]
    band_thresholds.append(1500)
    band_thresholds = np.array(band_thresholds).reshape(-1,1)
    activity_array = np.concatenate([flow.up_packets_length/band_thresholds,flow.down_packets_length/band_thresholds])
    # must transpose the array as flow has (numbands, timesteps)
    return activity_array.T


def maxNormalizeFlow(flow : FlowRepresentation):

        def normalizeArrayByBand(array : np.ndarray):
            # array is of shape (bands,timesteps), we divide timestamps by the max in each slot
            maxes = array.max(axis= -1,keepdims= True) + 1e-6
            array = array/maxes
            return array
    
        feature_array = np.concatenate((flow.up_packets,flow.down_packets,flow.up_bytes,flow.down_bytes),axis= 0)
        feature_array = normalizeArrayByBand(feature_array).T
        normalized_packetlengths = getActivityArrayFromFlow(flow= flow)
        feature_array = np.concatenate((feature_array,normalized_packetlengths), axis= 1)
        return feature_array


def minimizeOverlaps(starting_points,requested_interval,required_number_of_points):
    """
    Runs a binary search over max interval up untill requested interval
    so that we can get required_number_of_points while being as non overlapping as possible.
    """

    def pointsAfterNonOverlap(starting_points : List[int], interval_length) -> int:
        # starting points are sorted
        ans = 0
        last = starting_points[0] + interval_length
        included_starting_points = [starting_points[0]]
        for i in range(1,len(starting_points)):
            if starting_points[i] < last:
                ans += 1
            else:
                included_starting_points.append(starting_points[i])
                last = starting_points[i] + interval_length
        return included_starting_points
    
    if len(starting_points) == 0:
        return starting_points
    if len(starting_points) < required_number_of_points:
        # nonsense
        return starting_points
    
    
    starting_points.sort()
    mn_interval = 1
    mx_interval = requested_interval
    optimized_interval = None
    remaining_points_answer = []
    while mn_interval <= mx_interval:

        mid_interval = (mn_interval + mx_interval)//2
        remaining_points = pointsAfterNonOverlap(starting_points= starting_points,interval_length= mid_interval)

        if len(remaining_points) < required_number_of_points:
            mx_interval = mid_interval -1
        else:
            optimized_interval = mid_interval
            remaining_points_answer = remaining_points
            mn_interval = mid_interval + 1
    
    return remaining_points_answer
