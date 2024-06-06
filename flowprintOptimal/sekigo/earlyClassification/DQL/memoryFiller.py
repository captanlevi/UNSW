from ...flowUtils.flowDatasets import BaseFlowDataset
from .core import Rewarder, State, MemoryElement
from typing import List
import pandas as pd
import numpy as np
import random
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader


class MemoryFiller:
    def __init__(self,dataset,rewarder : Rewarder,min_length : int,max_length, ood_config, use_balancer : bool):
        self.dataset = dataset
        self.rewarder = rewarder

        self.min_length = min_length
        self.max_length = max_length

        self.actions = list(self.dataset.label_to_index.values())  
        self.actions.append(len(self.actions))

        self.ood_config = ood_config
        self.use_balancer = use_balancer
        if self.use_balancer == True and self.dataset.aug == None:
            print("Warning using balancer without aug !!!! not recommended but not tested yet")

    def processSingleSample(self,data):
        flow, label = data["data"], data["label"]
        memory_elements : List[MemoryElement] = []
        for length in range(self.min_length, self.max_length+1):
            for action in self.actions:
                state = State(timeseries= flow,label= label,length= length)
                reward, terminate = self.rewarder.reward(state= state,action= action)
                
                next_state = State(timeseries= flow,label= label,length= length + 1)
                if terminate == True:
                    # I am reducing the length as I will ahve to pass the state to LSTM 
                    # So I instead of filtering I will just zero all terminal states later.
                    next_state.length -= 1
                    next_state.setTerminal()

                
                memory_element = MemoryElement(state= state,action= action,reward= reward,next_state= next_state)
                memory_elements.append(memory_element)
        return memory_elements
    
    def getMemoryElements(self,is_ood):
        original_aug = self.dataset.aug
        if is_ood == True:
            self.dataset.aug = self.ood_config["ood_aug"]

        loader = None
        if self.use_balancer == False:
            loader = DataLoader(dataset= self.dataset,batch_size= 64, shuffle= True)
        else:
            loader = DataLoader(dataset= self.dataset,batch_size= 64,sampler= ImbalancedDatasetSampler(self.dataset))

        memory_elements = []
        for batch in loader:
            batch_data = batch["data"].numpy()
            batch_labels = batch["label"].numpy()
            for data,label in zip(batch_data,batch_labels):
                if is_ood == True and random.random() <= self.ood_config["ood_prob"]:
                    label = -1
                    memory_elements.extend(self.processSingleSample(dict(data = data, label = label)))
                else:
                    memory_elements.extend(self.processSingleSample(dict(data = data, label = label)))

        
        self.dataset.aug = original_aug
        return memory_elements
    

    def processDataset(self):
        memory_elements = []
        desired_memory_elements = len(self.dataset)*(self.max_length - self.min_length + 1)*len(self.actions)

        if self.ood_config != None:
            desired_memory_elements += int(self.ood_config["ood_prob"]*len(self.dataset))

        while len(memory_elements) <= desired_memory_elements:
            memory_elements.extend(self.getMemoryElements(is_ood= False))
            if self.ood_config != None:
                memory_elements.extend(self.getMemoryElements(is_ood= True))
        
        return memory_elements

                





        