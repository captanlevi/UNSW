import numpy as np


class State:
    def __init__(self,timeseries : np.ndarray, length,label : int):
        """
        The timeseries is of length (time_steps,num_features)
        We dont want to copy the the time series in the state every time so timeseries is a reference.
        The actual state is timeseries[:length]
        """
        self.timeseries = timeseries
        self.length = length
        self.label = label
        self.__is_terminal = False
    
    def setTerminal(self):
        self.__is_terminal = True
    
    def isTerminal(self):
        return self.__is_terminal
    

    
    


class MemoryElement:
    def __init__(self,state : State,action : int,reward : float,next_state : State):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state



class Rewarder:
    def __init__(self,max_length,l,num_labels : int):
        self.max_length = max_length
        self.l = l # l is smaller than 1
        self.num_labels = num_labels

    def reward(self,state : State,action : int):
        if state.label == action:
            # reward 1 on a correct prediction
            return 1, True
        else:
            # either incorrect or wait
            # wait 
            # treat the wait action with a negative reward
            if action == self.num_labels:
                if state.length == self.max_length:
                    # it is the last timestamp
                    return -self.l/self.max_length, True#-self.l*(state.length/self.max_length),True
                else:
                    return  -self.l/self.max_length, False#-self.l*(state.length/self.max_length), False
            else:
                # incorrect
                return -1,True