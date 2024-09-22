import numpy as np
from collections import deque
from .commons import getLabelMapping
class Grouper:
    """
    splits are like (a,b]
    """
    def __init__(self,features : list,labels : list, threshold = .01):
        assert len(features) == len(labels)
        assert threshold > 0
        self.label_mapping = getLabelMapping(labels= labels)
        labels = [self.label_mapping[x] for x in labels]
        self.labels = [label for _, label in sorted(zip(features, labels))]
        self.features = sorted(features)
        self.threshold = threshold
        self.splits = self.group()


    def getOptimalSplitIndex(self,start_index,end_index,value_counts : np.ndarray):
        # features are sorted
        #curr_entropy = Grouper.calcEntropy(np.array(labels))
        #if  curr_entropy == 0:
        #    assert False, "cant split perfect entropy array"
    

        indices_to_split_found = False
        
        right_value_count_array = value_counts
        left_value_count_array = np.zeros(len(self.label_mapping))

        min_entropy = float("inf")
        best_split = None
        best_split_counts = []
        for i in range(start_index,end_index):
            left_value_count_array[self.labels[i]] += 1
            right_value_count_array[self.labels[i]] -= 1
            if self.features[i] == self.features[i+1]:
                continue
            indices_to_split_found = True
            entropy = (Grouper.calcEntropyEfficient(value_counts_array= left_value_count_array) + Grouper.calcEntropyEfficient(value_counts_array= right_value_count_array))/2

            if entropy < min_entropy:
                min_entropy = entropy
                best_split = i
                best_split_counts = [left_value_count_array.copy(), right_value_count_array.copy()]
        
        if indices_to_split_found == False:
            return -1,None
        return best_split,best_split_counts
    
    
    def _getPossibleCutIndices(self,start_index,end_index):
        indices = []
        for i in range(start_index,end_index):
            if self.features[i] != self.features[i+1]:
                indices.append(i)
        return indices


    @staticmethod
    def calcEntropy(values : np.ndarray):
        # values is a 1D array
        _,counts = np.unique(values, return_counts= True)
        probs = counts/counts.sum()
        entropy = (-probs*np.log2(probs)).sum()
        return entropy
    
    @staticmethod
    def calcEntropyEfficient(value_counts_array : np.ndarray):
        """
        array [12,0,2,43,0] of counts of unique elements
        """
        probs = value_counts_array/value_counts_array.sum()
        non_zero_probs = probs[probs != 0]
        entropy =  -(non_zero_probs*np.log2(non_zero_probs)).sum()
        return entropy
    

    def splitInterval(self,interval,value_counts):
        assert interval[0] <= interval[1]
        interval_entropy =  Grouper.calcEntropy(values= self.labels[interval[0]: interval[1] + 1])
        if interval_entropy <= self.threshold:
            return None,None
        split_index, left_right_counts =  self.getOptimalSplitIndex(start_index= interval[0], end_index= interval[1],value_counts= value_counts)

        if split_index == -1:
            return None,None
        return split_index ,left_right_counts
        

    def split(self):
        value_counts = np.zeros(len(self.label_mapping))
        for label in self.labels:
            value_counts[label] += 1

        intervals = deque()
        intervals.append(([0,len(self.labels) - 1],value_counts))
        split_indices = []
        while len(intervals) != 0:
            curr_interval, value_counts = intervals.popleft()
           
            #splits = Parallel(n_jobs=min(12,len(intervals)))(delayed(self.splitInterval)(interval) for interval in intervals)    
            split_index, left_right_value_counts  = self.splitInterval(interval= curr_interval,value_counts = value_counts)

            if split_index == None:
                continue
            (left_value_count, right_value_count) = left_right_value_counts
            intervals.append(([curr_interval[0], split_index], left_value_count) )
            intervals.append(([split_index+1, curr_interval[1]], right_value_count))
            split_indices.append(split_index)


        
        return split_indices
        
      

    def getIntervals(self,split_indices):
        if len(split_indices) == 0:
            return [(-float("inf"), float("inf"))]

        split_indices.sort() # important
        discreet_intervals = [ (-float("inf"), self.features[split_indices[0]])]
        for i in range(1, len(split_indices)):
            feature_value = self.features[split_indices[i]]
            last_discreet_interval_end = discreet_intervals[-1][1]
            if feature_value != last_discreet_interval_end:
                discreet_intervals.append((last_discreet_interval_end, feature_value))
        
        last_discreet_interval_end = discreet_intervals[-1][1]
        discreet_intervals.append((last_discreet_interval_end , float("inf")))
        return discreet_intervals



    def group(self):
        split_indices = self.split()
        return self.getIntervals(split_indices= split_indices)

