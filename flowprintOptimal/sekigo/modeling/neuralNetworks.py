import torch.nn as nn
import torch
from torch.nn.utils.rnn import unpack_sequence
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    

    def save(self,model_path):
        torch.save(self.cpu().state_dict(), model_path)

    def load(self,model_path):
        self.load_state_dict(torch.load(model_path))


class BaseLSTMNetwork(NeuralNetwork):
    def __init__(self,lstm_input_size,lstm_hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size= lstm_input_size,hidden_size= lstm_hidden_size,batch_first= True)

    def _unpackAndGetFeatureFromLSTMOutput(self,lstm_out : nn.utils.rnn.PackedSequence):
        lstm_out = unpack_sequence(lstm_out)
        lstm_out = list(map(lambda x : x[-1],lstm_out))
        lstm_out = torch.stack(lstm_out,dim= 0)
        return lstm_out

class LSTMNetwork(BaseLSTMNetwork):
    def __init__(self,lstm_input_size,lstm_hidden_size,output_dim) -> None:
        super().__init__(lstm_hidden_size=lstm_hidden_size,lstm_input_size=lstm_input_size)
        self.output_dim = output_dim
        self.linear = nn.Linear(lstm_hidden_size,output_dim)
    
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
            lstm_out = self._unpackAndGetFeatureFromLSTMOutput(lstm_out= lstm_out)

        out = self.linear(lstm_out)
        return out#/torch.linalg.norm(out,dim = -1,keepdim= True)
    

    def earlyClassificationForward(self,X):
        """
        X is the timeseries input of shape 
        (BS,Seq len, lstm_input_size)
        outputs (BS,seq_len,feature_len)
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(X)
            return self.linear(lstm_out)


     
    



class TransformerGenerator(NeuralNetwork):
    def __init__(self,output_dim,random_dim,seq_len,embedding_dim,num_layers,num_heads,device,is_img,max_seq_len = 100) -> None:
        super().__init__()

        self.device = device
        self.random_dim = random_dim
        self.seq_len = seq_len
        self.is_img = is_img
        self.random_to_embedding_dim_linear = nn.Linear(random_dim,embedding_dim)
        self.positional_encodings = nn.Embedding(max_seq_len,embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding_dim_to_output_linear = nn.Sequential(nn.Linear(embedding_dim,output_dim),nn.Sigmoid())
        self.to(device)
    

    def forward(self,z):

        batch_size,seq_len ,_ = z.size()
        z = self.random_to_embedding_dim_linear(z)
        positions = torch.arange(0, seq_len).unsqueeze(0).expand([batch_size,-1]).to(self.device)
        x = z + self.positional_encodings(positions)

        x = self.transformer_encoder(x)
        x = self.embedding_dim_to_output_linear(x)
        if self.is_img:
            x = self.makeImageFromGeneratorOutput(x)
        return x
    
    def generateRandomZ(self,batch_size):
        return torch.randn(batch_size,self.seq_len,self.random_dim)
    

    def makeImageFromGeneratorOutput(self,gen_out):
        """
        gen_out is (BS,seq_len,patch_size)
        """
        batch_size,seq_len = gen_out.size(0),gen_out.size(1)
        square_size = np.sqrt(gen_out.shape[-1])
        assert square_size%1 == 0
        square_size = int(square_size)
        gen_out = gen_out.view(batch_size,seq_len,square_size,square_size)
        num_squares = np.sqrt(seq_len)
        assert num_squares%1 == 0
        num_squares = int(num_squares)
        image_dim = num_squares*square_size
        img = torch.empty(batch_size,image_dim,image_dim).to(self.device)


        index = 0
        for i in range(0,image_dim,square_size):
            for j in range(0,image_dim,square_size):
                img[:,i:i+square_size,j:j+square_size] = gen_out[:,index]
                index += 1
        
        img = img.unsqueeze(1)
        return img
        




class TransformerFeatureGenerator(nn.Module):
    def __init__(self,random_dim,embedding_dim,num_layers,num_heads,device,max_seq_len = 100) -> None:
        super().__init__()

        self.device = device
        self.random_dim = random_dim
        self.random_to_embedding_dim_linear = nn.Linear(random_dim,embedding_dim)
        self.positional_encodings = nn.Embedding(max_seq_len,embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding_dim_to_ts_dim_linear = nn.Sequential(nn.Linear(embedding_dim,1),nn.Sigmoid())

    

    def forward(self,z):
        """
        Takes in z as input (BS,N,random_dim)
        returns feature vector (BS,feature_dim,1) but i return (BS,feature_dim)
        """

        batch_size,seq_len ,_ = z.size()
        z = self.random_to_embedding_dim_linear(z)
        positions = torch.arange(0, seq_len).unsqueeze(0).expand([batch_size,-1]).to(self.device)
        x = z + self.positional_encodings(positions)

        x = self.transformer_encoder(x)
        x = self.embedding_dim_to_ts_dim_linear(x)[:,:,0]
        return x


class CNNNetwork(NeuralNetwork):
    def __init__(self, ts_dim, num_filters, kernel_sizes,output_dim):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=ts_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=num_filters, out_channels=2*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=2*num_filters, out_channels=4*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
                #nn.MaxPool1d(kernel_size=2)
            )
            self.conv_blocks.append(conv_block)
        
        
        self.fc = nn.Linear(4*num_filters * len(kernel_sizes), output_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, ts_dim) 
        # Apply convolutional blocks
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_output = conv_block(x.permute(0, 2, 1).contiguous())  # Conv1D expects (batch_size, in_channels, seq_length)
            conv_output = F.max_pool1d(conv_output, kernel_size=conv_output.size(2)).squeeze(2)  # Global Max Pooling
            conv_outputs.append(conv_output)
        
        # Concatenate convolutional outputs

        conv_output_concat = torch.cat(conv_outputs, dim=1)
        return self.fc(conv_output_concat)
    




class CNNNetwork1D(NeuralNetwork):
    def __init__(self, ts_dim, num_filters, kernel_sizes,output_dim):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=ts_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=num_filters, out_channels=2*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=2*num_filters, out_channels=4*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=4*num_filters, out_channels=4*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.AdaptiveMaxPool1d(1),
                nn.Conv1d(in_channels= 4*num_filters,out_channels=1,kernel_size= 1)
                #nn.MaxPool1d(kernel_size=2)
            )
            self.conv_blocks.append(conv_block)
        
        
        self.fc = nn.Linear(len(kernel_sizes), output_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, ts_dim) 
        # Apply convolutional blocks
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_output = conv_block(x.permute(0, 2, 1).contiguous())[:,:,0]  # Conv1D expects (batch_size, in_channels, seq_length)
            #conv_output = F.max_pool1d(conv_output, kernel_size=conv_output.size(2)).squeeze(2)  # Global Max Pooling
            conv_outputs.append(conv_output)
        
        # Concatenate convolutional outputs

        conv_output_concat = torch.cat(conv_outputs, dim=1)
        return self.fc(conv_output_concat)




class CNNNetwork2D(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_sizes, output_dim):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=num_filters, out_channels=2*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=2*num_filters, out_channels=4*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=4*num_filters, out_channels=4*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.AdaptiveMaxPool2d(1),
                nn.Conv2d(in_channels= 4*num_filters, out_channels= output_dim, kernel_size= 1)
            )
            self.conv_blocks.append(conv_block)
        
        self.fc = nn.Linear(len(kernel_sizes)*output_dim, output_dim)

    def forward(self, x):
        """
        x is  (BS,num_channels,H,W)
        """
        # Apply convolutional blocks
        conv_outputs = []
        for conv_block in self.conv_blocks:
            
            conv_output = conv_block(x.contiguous())[:,:,0,0]
            conv_outputs.append(conv_output)


        conv_output_concat = torch.cat(conv_outputs, dim=1)
        return self.fc(conv_output_concat)




    

class LinearPredictor(nn.Module):
    def __init__(self,feature_dim,num_classes) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(feature_dim,feature_dim*2), nn.LeakyReLU(), nn.Linear(feature_dim*2,feature_dim*2), nn.LeakyReLU(),
                                    nn.Linear(feature_dim*2,feature_dim*2), nn.LeakyReLU(),nn.Linear(feature_dim*2,feature_dim), nn.LeakyReLU(),
                                    nn.Linear(feature_dim,num_classes)
                                    )
    
    def forward(self,X):
        return self.linear(X)
    

class Predictor(NeuralNetwork):
    def __init__(self,feature_dim,num_classes) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim,num_classes)
    def forward(self,X):
        return self.linear(X)