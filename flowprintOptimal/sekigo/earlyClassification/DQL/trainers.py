import torch
import torch.nn as nn
from ...modeling.neuralNetworks import LSTMNetwork, LinearPredictor
from .datasets import MemoryDataset
from ...flowUtils.flowDatasets import BaseFlowDataset
from copy import deepcopy
from torch.utils.data import DataLoader
from ...modeling.loggers import Logger
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from ...utils.commons import augmentData

class EarlyClassificationtrainer:
    def __init__(self,feature_extractor : LSTMNetwork,predictor : LinearPredictor,train_dataset : BaseFlowDataset,memory_dataset : MemoryDataset,
                 test_dataset : BaseFlowDataset,ood_dataset : BaseFlowDataset,logger : Logger,model_replacement_steps : int,device : str):
        
        self.device = device
        self.feature_extractor  = feature_extractor.to(device)
        self.lag_feature_extractor = deepcopy(feature_extractor).to(device)
        self.lag_feature_extractor.eval()

        self.predictor = predictor.to(device)
        self.lag_predictor = deepcopy(predictor).to(device)
        self.lag_predictor.eval()

        self.train_dataset = train_dataset
        self.memory_dataset = memory_dataset
        self.test_dataset = test_dataset
        self.ood_dataset = ood_dataset
        self.logger = logger

        self.mse_loss_function = nn.MSELoss()
        self.model_replacement_steps = model_replacement_steps

        self.logger.setMetricReportSteps(metric_name= "test_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "test_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval", step_size= 1)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    
    def trainFullClassifier(self,batch_size,feature_extractor_optimizer, predictor_optimizer):
        indices = np.random.randint(0,len(self.train_dataset),size= batch_size).tolist()
        X,y = [],[]

        for i in indices:
            data_point = self.train_dataset[i]
            data,label = data_point["data"], data_point["label"]
            X.append(data)
            y.append(label)
        
        # making augmented batch
        for i in indices[:batch_size//2]:
            data = augmentData(self.train_dataset[i]["data"])
            label = len(self.train_dataset.label_to_index)
            X.append(data)
            y.append(label)
        
        X,y = np.array(X),np.array(y)
        X = torch.tensor(X).float().to(self.device)
        y = torch.tensor(y).to(self.device).long()


        feature_extractor_optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        model_out = self.predictor(self.feature_extractor(X))
        loss = self.cross_entropy_loss(model_out,y)
        loss.backward()
        feature_extractor_optimizer.step()
        predictor_optimizer.step()
        self.logger.addMetric(metric_name= "full_classifier_loss", value= loss.cpu().item())



    def trainStep(self,steps,batch : dict,lam : float,feature_extractor_optimizer,predictor_optimizer):
        """
        state and next state is (BS,num_classes)
        """
        self.trainFullClassifier(batch_size= batch["action"].shape[0],feature_extractor_optimizer= feature_extractor_optimizer,predictor_optimizer= predictor_optimizer)


        state,next_state,action,reward,is_terminal = batch["state"].to(self.device), batch["next_state"].to(self.device),\
                                                    batch["action"].to(self.device), batch["reward"].to(self.device),batch["is_terminal"].to(self.device)
        with torch.no_grad():
            next_state_max_actions_model = torch.argmax(self.predictor(self.feature_extractor(next_state)),dim = -1,keepdim= True)
            next_state_values_lag_model = self.lag_predictor(self.lag_feature_extractor(next_state))
            next_state_values_for_max_action = torch.gather(input= next_state_values_lag_model, dim= 1, index= next_state_max_actions_model)
            next_state_values_for_max_action = next_state_values_for_max_action*(~(is_terminal.unsqueeze(-1)))
            target = reward + lam*(next_state_values_for_max_action.squeeze())
        
        predicted_values = self.predictor(self.feature_extractor(state))
        predicted_values_for_taken_action = torch.gather(input= predicted_values, dim= 1,index= action.unsqueeze(-1)).squeeze()
        
        loss = self.mse_loss_function(target, predicted_values_for_taken_action.squeeze()).mean()

        loss.backward()
        feature_extractor_optimizer.step()
        feature_extractor_optimizer.zero_grad()
        predictor_optimizer.step()
        predictor_optimizer.zero_grad()
        self.logger.addMetric(metric_name= "loss", value= loss.item())


        if steps%self.model_replacement_steps == 0:
            self.__refreshLagModel()
                
        return loss
        
    def __refreshLagModel(self):
        self.lag_feature_extractor = deepcopy(self.feature_extractor)
        self.lag_predictor = deepcopy(self.predictor)
        self.lag_predictor.eval()
        self.lag_feature_extractor.eval()
    

    def predictStep(self,batch_X):
        """
        Here the return is of shape (BS,seq_len)
        """
        self.predictor.eval()
        self.feature_extractor.eval()
        with torch.no_grad():
            model_out = self.predictor(self.feature_extractor.earlyClassificationForward(batch_X))
        self.predictor.train()
        self.feature_extractor.train()
        return torch.argmax(model_out,dim= -1)


    def __processSinglePrediction(self,prediction,num_classes):
        """
        predictions are of shape (seq_len)
        """
        
        for time in range(self.memory_dataset.min_length,len(prediction)):
            if prediction[time] < num_classes:
                return (prediction[time],time + 1)
        
        return (-1,len(prediction))
        
    def predictOnDataset(self,dataset : BaseFlowDataset):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        self.feature_extractor.eval()
        self.predictor.eval()
        with torch.no_grad():

            labels = []
            predictions = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(self.device), batch["label"].to(self.device)
                predicted = self.predictStep(batch_X = batch_X).cpu().numpy() # (BS,seq_len)
                processed_predictions = map(lambda x : self.__processSinglePrediction(x,len(dataset.label_to_index)), predicted)

                predictions.extend(processed_predictions)
                labels.extend(batch_y.cpu().numpy().tolist())

        self.feature_extractor.train()
        self.predictor.train()
        return predictions


    def eval(self,dataset : BaseFlowDataset):
        predictions = self.predictOnDataset(dataset= dataset)
        labels = list(map(lambda x : x["label"], dataset))

        predicted_labels,time = [],[]

        for i in range(len(predictions)):
            predicted_labels.append(predictions[i][0])
            time.append(predictions[i][1])
        _,_,f1,_ = precision_recall_fscore_support(labels, predicted_labels, average= "weighted",zero_division=0)
        average_time = sum(time)/len(time)
        return f1,average_time
    

    def evalTrain(self):
        f1,average_time = self.eval(dataset= self.train_dataset)
        self.logger.addMetric(metric_name= "train_eval_f1", value= f1)
        self.logger.addMetric(metric_name= "train_eval_time", value= average_time)

    def evalTest(self):
        f1,average_time = self.eval(dataset= self.test_dataset)
        self.logger.addMetric(metric_name= "test_eval_f1", value= f1)
        self.logger.addMetric(metric_name= "test_eval_time", value= average_time)

    def evalOOD(self):
        predictions = self.predictOnDataset(self.ood_dataset)
        predictions = list(map(lambda x : x[0],predictions))
        predictions = np.array(predictions)
        accuracy = (predictions == -1).sum()/predictions.shape[0]
        self.logger.addMetric(metric_name= "ood_eval", value= accuracy)
        

        

    def train(self,epochs : int,batch_size = 64,lr = .001,lam = .99,model_lag_in_steps = 50):
        # TODO add batch_sampler
        """
        Can stress enough how important the shuffle == True is in the Dataloader
        """
        train_dataloader = DataLoader(dataset= self.memory_dataset,collate_fn= self.memory_dataset.collateFn,batch_size= batch_size,drop_last= True,shuffle= True)
        
        feature_extractor_optimizer = torch.optim.Adam(params= self.feature_extractor.parameters(), lr= lr)
        predictor_optimizer = torch.optim.Adam(params= self.predictor.parameters(), lr = lr)
        steps = 0

        for epoch in range(epochs):
            for batch in train_dataloader:
                self.trainStep(steps = steps,batch= batch,lam= lam, feature_extractor_optimizer= feature_extractor_optimizer, predictor_optimizer= predictor_optimizer)
                steps += 1            

                if steps%1000 == 0:
                    self.evalTest()
                    self.evalOOD()
                if steps%2000 == 0:
                    self.evalTrain()
                


