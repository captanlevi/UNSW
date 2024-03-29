import torch.nn as nn
import torch
from torch.nn.utils.rnn import unpack_sequence

class LSTMClassifier(nn.Module):
    def __init__(self,lstm_input_size,lstm_hidden_size,num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size= lstm_input_size,hidden_size= lstm_hidden_size,batch_first= True)
        self.linear =nn.Sequential( nn.Linear(in_features= lstm_hidden_size,out_features = lstm_hidden_size//2), nn.ReLU(), nn.Linear(lstm_hidden_size//2,num_classes))
        self.softmax = nn.Softmax(dim= -1)
        self.loss_function = nn.CrossEntropyLoss()
        self.mse_loss_function = nn.MSELoss()
        # TODO look at the projection size

    def __unpackAndGetFeatureFromLSTMOutput(self,lstm_out : nn.utils.rnn.PackedSequence):
        lstm_out = unpack_sequence(lstm_out)
        lstm_out = list(map(lambda x : x[-1],lstm_out))
        lstm_out = torch.stack(lstm_out,dim= 0)
        return lstm_out
        
    
    def forward(self,X):
        """
        X is the timeseries input of shape 
        (BS,Seq len, lstm_input_size)
        OR
        packed_sequence

        The output is of shape (BS,num_classes)
        """
        lstm_out, _ = self.lstm(X)
        if isinstance(X,torch.Tensor):
            lstm_out = lstm_out[:,-1,:]
        else:
            lstm_out = self.__unpackAndGetFeatureFromLSTMOutput(lstm_out= lstm_out)
        return self.linear(lstm_out)
    
    def earlyClassificationForward(self,X):
        """
        X is the timeseries input of shape 
        (BS,Seq len, lstm_input_size)
        outputs (BS,seq_len,num_classes)
        """

        with torch.no_grad():
            lstm_out, _ = self.lstm(X)
            return self.linear(lstm_out)


    
    def calculateLoss(self,predicted,targets):
        return self.loss_function(predicted,targets).mean()
