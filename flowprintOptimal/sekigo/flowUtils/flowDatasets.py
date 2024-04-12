from typing import List
from ..core.flowRepresentation import FlowRepresentation
import datetime
from torch.utils.data import Dataset
from .commons import getActivityArrayFromFlow, maxNormalizeFlow
import numpy as np


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


class ActivityDataset(BaseFlowDataset):
    def __init__(self,flows : List[FlowRepresentation],label_to_index : dict):
        super().__init__(flows= flows,label_to_index= label_to_index)
    
    def __getitem__(self, index) -> FlowRepresentation:
        return dict(data = getActivityArrayFromFlow(self.flows[index]), label = self.label_to_index[self.flows[index].class_type])

    @staticmethod
    def collateFn():
        pass


class DDQNActivityDataset(BaseFlowDataset):
    def __init__(self, flows: List[FlowRepresentation], label_to_index: dict):
        super().__init__(flows = flows, label_to_index= label_to_index)

        self.labels = list(map(lambda x : self.label_to_index[x.class_type],self.flows))
        self.flows = list(map(lambda x : getActivityArrayFromFlow(x), self.flows))
    
    def __getitem__(self, index):
        return dict(data = self.flows[index], label  = self.labels[index])


class MaxNormalizedDataset(BaseFlowDataset):
    def __init__(self,flows : List[FlowRepresentation],label_to_index : dict):
        super().__init__(flows= flows,label_to_index= label_to_index)
    
    def __getitem__(self, index) -> FlowRepresentation:
        return dict(data = maxNormalizeFlow(self.flows[index]), label = self.label_to_index[self.flows[index].class_type])
    
