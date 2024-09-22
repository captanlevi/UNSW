from .grouper import Grouper
import numpy as np
from .commons import getLabelMapping, loadLVectors
import json


class LVector:
    def __init__(self, lengths, directions, forward_grouper : Grouper, backward_grouper : Grouper, class_type):
        """
        Looks like [(v,->), (v,<-), ......]
        where v is value and arrows are direction

        where 0 is forward direction and 1 is backward
        """
        assert len(lengths) == len(directions)
        
        single_bin_splits = [(-float("inf"), float("inf"))]
        self.lv1 = self.encode(lengths= lengths, directions= directions, forward_split= None, backward_split= None)
        self.lv2 = self.encode(lengths= lengths, directions= directions, forward_split= forward_grouper.splits, backward_split= backward_grouper.splits)
        self.lv3 = self.encode(lengths= lengths, directions= directions,forward_split= forward_grouper.splits, backward_split= single_bin_splits)
        self.lv4 = self.encode(lengths= lengths, directions= directions, forward_split= single_bin_splits, backward_split= backward_grouper.splits)
        self.lv5 = self.encode(lengths= lengths, directions= directions, forward_split= single_bin_splits, backward_split= single_bin_splits)

        self.class_type = class_type

    def encode(self,lengths,directions,forward_split,backward_split):
        
        encoding = []

        for l,d in zip(lengths,directions):
            if d == 0:
                if forward_split == None:
                    encoding.append( (l,"->"))
                else:
                    encoding.append((LVector.assignSplit(value= l,splits= forward_split),"->" ))
            elif d == 1:
                if backward_split == None:
                    encoding.append( (l,"<-"))
                else:
                    encoding.append((LVector.assignSplit(value= l,splits= backward_split),"<-" ))
            else:
                assert False, "direction can be either 1 or 0"
        return encoding
        

    @staticmethod
    def assignSplit(value,splits):
        # split values are (a,b]
        for i in range(len(splits)):
            curr_split = splits[i]

            if curr_split[0] < value <= curr_split[1]:
                return i
        
        assert False, "split not complete, {}".format(splits)


    
        
        

class Snippet:
    def __init__(self, sequence, position, negation, tp : int):
        """
        tp defines what type of l vector the snippet is supposed to match
        there are 5 types [1,2,3,4,5]

        position starts from 1, so to get "index" substract 1
        """
        assert tp in [1,2,3,4,5], "wrong tp"
        self.sequence = sequence
        self.position = position
        self.negation = negation
        self.tp = tp    

    def __lt__(self,other):
        """
        Dummy le function for heap comparision
        """
        return True


    def __match(self,encoding, start_index):
        assert start_index >= 0
        encoding_match_length = len(encoding) - start_index
        if encoding_match_length < len(self.sequence):
            return False
        
        for i in range(len(self.sequence)):
         
            if self.sequence[i] != encoding[start_index + i]:
                return False
        return True
    def _matchLeftAnchored(self,encoding):
        assert self.position > 0
        return self.__match(encoding= encoding, start_index= self.position - 1)
        
    
    def _matchRightAnchored(self,encoding):
        assert self.position < 0
        start_index = len(encoding) - (-self.position) 
        if start_index < 0:
            # length of the encoding is smaller than the position from the end
            return False
        return self.__match(encoding= encoding, start_index= start_index)
    
    def _matchUnAnchored(self,encoding):
        assert self.position == "*"
        for start_index in range(0,len(encoding) - len(self.sequence) + 1):
            if self.__match(encoding= encoding, start_index= start_index) == True:
                return True
        return False

    

    def match(self,lvector : LVector):
        encoding = None
        if self.tp == 1:
            encoding = lvector.lv1
        elif self.tp == 2:
            encoding = lvector.lv2
        elif self.tp == 3:
            encoding = lvector.lv3
        elif self.tp == 4:
            encoding = lvector.lv4
        else:
            encoding = lvector.lv5
        
        if len(encoding) < len(self.sequence):
            return False
        

        if self.position == "*":
            return self._matchUnAnchored(encoding= encoding)
        elif self.position > 0:
            return self._matchLeftAnchored(encoding= encoding)
        else:
            return self._matchRightAnchored(encoding= encoding)
        

        


class SnippetScorer:
    def __init__(self,l_vectors):
        if isinstance(l_vectors, str):
            self.l_vectors = loadLVectors(path= l_vectors)
        else:
            self.l_vectors = l_vectors

        
        labels = list(map(lambda x : x.class_type, self.l_vectors))
        self.label_mapping = getLabelMapping(labels= labels)
        self.labels = [self.label_mapping[x] for x in labels]
        self.label_counts = np.zeros(len(self.label_mapping))
        for label in self.labels:
            self.label_counts[label] += 1


    def score(self,snippet : Snippet):
        matches = [snippet.match(l_vector) for l_vector in self.l_vectors]
        #matches = Parallel(n_jobs=min(len(self.l_vectors), 12))(delayed(snippet.match)(l_vector) for l_vector in self.l_vectors)
        label_match_array = np.zeros(len(self.label_mapping))
        for i in range(len(matches)):
            if matches[i] == True:
                label_match_array[self.labels[i]] += 1


        # now getting scores
        score_a = np.log((1 + label_match_array)/self.label_counts) 
        score_b = np.zeros_like(score_a)

        for i in range(len(self.label_mapping)):
            total_non_class_weight = self.label_counts.sum() - self.label_counts[i]
            matched_non_class_weight = label_match_array.sum() - label_match_array[i]
            score_b[i] = -np.log((matched_non_class_weight + 1)/total_non_class_weight)

        score = score_a + score_b
        return score.max()