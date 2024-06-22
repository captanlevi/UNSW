from typing import List
from ..core.flowRepresentation import FlowRepresentation, TimeslotRepresentation
import datetime
from torch.utils.data import Dataset
from .commons import getActivityArrayFromFlow, maxNormalizeFlow , getActivityArrayFromTimeslotRep
import numpy as np
import random
from ..utils.commons import augmentData
from torchsampler import ImbalancedDatasetSampler


class BaseFlowDataset(Dataset):

    def __init__(self,flows : List[FlowRepresentation],label_to_index : dict):
        """
        Its very important to give the label_to_index while creating a test dataset
        so its the same for train and test
        """
        super().__init__
        self.flows = flows
        self.label_to_index = self.__getLabelDict() if label_to_index == None else label_to_index
        self.flow_config = flows[0].flow_config
        
    
    def __getLabelDict(self):
        # do not change this function the labels must of zero indixed if not it will break the DDQN training
        label_to_index = dict()
        counter = 0

        for flow in self.flows:
            class_type = flow.class_type
            if class_type not in label_to_index:
                label_to_index[class_type] = counter
                counter += 1
        
        return label_to_index
        
    
    def __len__(self):
        return len(self.flows)
    
    def get_labels(self):
        return list(map(lambda x : self.label_to_index[x.class_type],self.flows))


class ActivityDataset(BaseFlowDataset):
    def __init__(self,flows : List[FlowRepresentation],label_to_index : dict):
        super().__init__(flows= flows,label_to_index= label_to_index)
    
    def __getitem__(self, index) -> FlowRepresentation:
        return dict(data = getActivityArrayFromFlow(self.flows[index]), label = self.label_to_index[self.flows[index].class_type])

    @staticmethod
    def collateFn():
        pass


class DDQNActivityDataset(BaseFlowDataset):
    def __init__(self, flows: List[FlowRepresentation| TimeslotRepresentation], label_to_index: dict,aug = None):
        super().__init__(flows = flows, label_to_index= label_to_index)
        self.aug = aug
        self.labels = list(map(lambda x : self.label_to_index[x.class_type],self.flows))
        if isinstance(flows[0], TimeslotRepresentation):
            self.flows =  list(map(lambda x : getActivityArrayFromTimeslotRep(x), self.flows))
        else:
            self.flows = list(map(lambda x : getActivityArrayFromFlow(x), self.flows))
    
    def __getitem__(self, index):
        return dict(data = self.flows[index] if (self.aug == None) else augmentData(self.flows[index],fraction_range= self.aug), label  = self.labels[index])
    
    def get_labels(self):
        return self.labels







        
    

class MaxNormalizedDataset(BaseFlowDataset):
    def __init__(self,flows : List[FlowRepresentation],label_to_index : dict):
        super().__init__(flows= flows,label_to_index= label_to_index)
    
    def __getitem__(self, index) -> FlowRepresentation:
        return dict(data = maxNormalizeFlow(self.flows[index]), label = self.label_to_index[self.flows[index].class_type])
    


class PacketFlowDataset(Dataset):
    def __init__(self,flows,label_to_index,aug = None):
        # aug is a range of augmentation on such good for training is [0,.4]
        super().__init__()
        self.flows = flows
        self.aug = aug
        if label_to_index == None:
            self.label_to_index = self.__getLabelDict()
        else:
            self.label_to_index = label_to_index


    
    def getDataItem(self,index,aug):
        flow = self.flows[index]
        data = np.array([flow.lengths,flow.inter_arrival_times,flow.directions]).T
        return self._augmentData(data= data, aug_lims= aug) if aug != None else data



    def __len__(self):
        return len(self.flows)
    
    def __getitem__(self, index):
        flow = self.flows[index]
        data = self.getDataItem(index= index, aug= self.aug)
        return dict(data = data,label = self.label_to_index[flow.class_type])

    def __getLabelDict(self):
        # do not change this function the labels must of zero indixed if not it will break the DDQN training
        label_to_index = dict()
        counter = 0

        for flow in self.flows:
            class_type = flow.class_type
            if class_type not in label_to_index:
                label_to_index[class_type] = counter
                counter += 1
        
        return label_to_index
    

    def get_labels(self):
        return list(map(lambda x : self.label_to_index[x.class_type],self.flows))
    

    def _augmentData(self,data,aug_lims = [.1,.6]):
        """
        (TS,3) last column is direction
        PLZ bhai datacopy kar le usko modify mat kar inplace
        """
        TS = data.shape[0]
        
        aug_data = data.copy()
        all_indices = np.arange(TS)
        for i in range(3):
            aug_fraction = aug_lims[0] + (aug_lims[1] - aug_lims[0])*np.random.random()
            count = int(TS*aug_fraction)
            indices = np.random.choice(a= all_indices,size= count,replace= False)
            if i < 2:
                # in case of inter_arrival_time and payload size we replace with random values from 0 to 1
                aug_data[indices,i] = np.random.random(count)
            else:
                # in case of direction we switch the direction or should I replace with random 0s and ones ?
                aug_data[indices,i] = 1 - aug_data[indices,i]

        return  aug_data