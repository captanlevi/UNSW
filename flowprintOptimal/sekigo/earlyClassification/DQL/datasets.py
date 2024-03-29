from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .core import MemoryElement
from typing import List
import torch
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

class MemoryDataset(Dataset):
    """
    This dataset is used for internal training for the DQN 
    """
    def __init__(self, memories : List[MemoryElement],num_classes : int):
        super().__init__()
        self.memories = memories
        self.num_classes = num_classes
    

    def __getitem__(self, index : int):
        memory = self.memories[index]
        state_length,next_state_length = memory.state.length, memory.next_state.length
        return dict(
            state = memory.state.timeseries[:state_length,:],
            next_state = memory.next_state.timeseries[:next_state_length,:],
            is_terminal = memory.next_state.isTerminal(),
            reward = memory.reward,
            action = memory.action
        )
    

    def __getActions(self):
        return list(map(lambda x : x.action,self.memories))

        
    def __len__(self):
        return len(self.memories)
    

    def collateFn(self,batch):   
        state = list(map(lambda x : torch.tensor(x["state"]).float(),batch ))
        next_state = list(map(lambda x : torch.tensor(x["next_state"]).float(), batch))

        state = pack_sequence(state,enforce_sorted= False)
        next_state = pack_sequence(next_state,enforce_sorted= False)


        is_terminal = torch.tensor(list(map(lambda x : x["is_terminal"],batch))).bool()
        reward = torch.tensor(list(map(lambda x : x["reward"],batch))).float()
        action = torch.tensor(list(map(lambda x : x["action"],batch))).long()

        return dict(
            state = state, next_state = next_state, action = action, reward = reward,is_terminal = is_terminal
        )
    
    def getSampler(self):
        weights = [1]*self.num_classes
        weights[-1] = self.num_classes -1 
        action_to_weights = dict()
        for i in range(self.num_classes):
            action_to_weights[i] = weights[i]
        actions = self.__getActions()

        sample_weights = list(map(lambda x : action_to_weights[x], actions))
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples= len(sample_weights), replacement=True)
        return sampler
    

    def getWeightedDataloader(self,batch_size ):
        sampler = self.getSampler()
        loader = DataLoader(dataset= self,collate_fn= self.collateFn,sampler= sampler,batch_size= batch_size)
        return loader




    