from ...flowUtils.flowDatasets import BaseFlowDataset
from .core import Rewarder, State, MemoryElement
from typing import List
import numpy as np

class MemoryFiller:
    def __init__(self,dataset : BaseFlowDataset,rewarder : Rewarder,min_length_in_seconds : int,max_length_in_seconds,random_sample_ratio : int):
        self.dataset = dataset
        self.rewarder = rewarder

        grain = self.dataset.flow_config.grain
        self.min_length = int(min_length_in_seconds/grain)
        self.max_length = int(max_length_in_seconds/grain)

        self.actions = list(self.dataset.label_to_index.values())  
        self.random_sample_ratio = random_sample_ratio
        self.actions.append(len(self.actions))

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
    def processDataset(self):

        memory_elements = []
        
        for i in range(1,len(self.dataset)+1):
            # using 1 to len(dataset) + 1 so modulo does not throw an error
            if self.random_sample_ratio%i == 0:
                random_data_value = np.random.random(self.dataset[i-1]["data"].shape)
                random_data_label = -1
                data = dict(data = random_data_value,label = random_data_label)
                memory_elements.extend(self.processSingleSample(data))

            data = self.dataset[i-1]
            memory_elements.extend(self.processSingleSample(data))
 
        return memory_elements

                





        