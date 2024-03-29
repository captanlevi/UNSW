from ...flowUtils.flowDatasets import BaseFlowDataset
from .core import Rewarder, State, MemoryElement
from typing import List

class MemoryFiller:
    def __init__(self,dataset : BaseFlowDataset,rewarder : Rewarder,min_length_in_seconds : int,max_length_in_seconds):
        self.dataset = dataset
        self.rewarder = rewarder

        grain = self.dataset.flow_config.grain
        self.min_length = int(min_length_in_seconds/grain)
        self.max_length = int(max_length_in_seconds/grain)

        self.actions = list(self.dataset.label_to_index.values())  
        self.actions.append(len(self.actions))

    def processSingleSample(self,data):
        flow, label = data["flow"], data["label"]
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
        
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            memory_elements.extend(self.processSingleSample(data))
 
        return memory_elements

                



class StrongConsistencyMemoryFiller:
    def __init__(self,dataset : BaseFlowDataset,rewarder : Rewarder,min_length_in_seconds : int,max_length_in_seconds):
        self.dataset = dataset
        self.rewarder = rewarder

        grain = self.dataset.flow_config.grain
        self.min_length = int(min_length_in_seconds/grain)
        self.max_length = int(max_length_in_seconds/grain)

        self.actions = list(self.dataset.label_to_index.values())  
        self.actions.append(len(self.actions))

    def processSingleSample(self,data):
        flow, label = data["flow"], data["label"]
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
        
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            memory_elements.extend(self.processSingleSample(data))
 
        return memory_elements
                

        