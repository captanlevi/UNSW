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
from ...utils.evaluations import EarlyEvaluation
from torch.utils.data import WeightedRandomSampler
import random



class EarlyClassificationtrainer:
    def __init__(self,predictor : LSTMNetwork,train_dataset : BaseFlowDataset,memory_dataset : MemoryDataset,hint_loss_alpha : float,q_loss_alpha : float,use_sampler : bool, hint_loss_gap : float,
                 test_dataset : BaseFlowDataset,ood_dataset : BaseFlowDataset,logger : Logger,model_replacement_steps : int,device : str):
        
        self.device = device
        self.use_sampler = use_sampler
        self.predictor = predictor.to(device)
        self.lag_predictor = deepcopy(predictor).to(device)
        self.lag_predictor.eval()

        self.hint_loss_alpha = hint_loss_alpha
        self.q_loss_alpha = q_loss_alpha
        self.hint_loss_gap = hint_loss_gap

        self.train_dataset = train_dataset
        self.memory_dataset = memory_dataset
        self.test_dataset = test_dataset
        self.ood_dataset = ood_dataset
        self.logger = logger

        self.best = dict(
            score = 0,
            model = deepcopy(self.predictor)
        )

        self.evaluator = EarlyEvaluation(min_steps= memory_dataset.min_length, device= device,model= self.predictor)

        self.mse_loss_function = nn.MSELoss(reduction= "none")
        self.model_replacement_steps = model_replacement_steps

        self.logger.setMetricReportSteps(metric_name= "test_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "test_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "incorrect_ood_test", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "incorrect_ood_train", step_size= 1)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction= "none")



    def getWeightedSampler(self,memory_dataset):
        """
        memory dataset has all actions at least once
        """
        actions = []

        for d in memory_dataset.memories:
            actions.append(d.action)
        actions = np.array(actions)

        unique_actions = np.unique(actions)
        weights = np.array([1/len(actions)]*len(actions))

        weights[actions == (len(unique_actions) - 1)] *= (len(unique_actions) -1)
        weights = weights/weights.sum()
        sampler = WeightedRandomSampler(weights= weights,num_samples= len(weights))
        return sampler
    
    def trainFullClassifier(self,batch_size, predictor_optimizer):
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


        predictor_optimizer.zero_grad()
        model_out = self.predictor(X)[0]
        loss = self.cross_entropy_loss(model_out,y)
        loss.backward()
        predictor_optimizer.step()
        self.logger.addMetric(metric_name= "full_classifier_loss", value= loss.cpu().item())


    def hintLoss(self,model_out,labels,state_lengths):
        """
        applying on predictions that are not waiting and are wrong only
        """
        wait_label = model_out.shape[1] - 1
        labels[labels == -1] = wait_label


        # editing the 
        mx_values,model_out_preds = torch.max(model_out,dim= -1) # (BS,)
        
        mask = ((model_out_preds != wait_label) & (model_out_preds != labels)).float()
        #weights = state_lengths/self.memory_dataset.max_length
        #self.mse_loss_function(mx_values, model_out[:,-1])
        model_correct_label_outputs = torch.gather(input= model_out,index= labels.unsqueeze(-1), dim= -1)[:,0]
        #model_wait_label_outputs = model_out[:,-1]

        #wait_greater_mask = (model_wait_label_outputs >= model_correct_label_outputs).float()

        hint_loss =  (mx_values - model_correct_label_outputs) + self.hint_loss_gap  # this is a gap parameter
        hint_loss = hint_loss*mask
    
        return hint_loss.sum()/(mask.sum() + 1e-8)
        


    def trainStep(self,steps,batch : dict,lam : float,predictor_optimizer):
        """
        state and next state is (BS,num_classes)
        """
        #self.trainFullClassifier(batch_size= batch["action"].shape[0],predictor_optimizer= predictor_optimizer)


        state,next_state,action,reward,is_terminal = batch["state"].to(self.device), batch["next_state"].to(self.device),\
                                                    batch["action"].to(self.device), batch["reward"].to(self.device),batch["is_terminal"].to(self.device)
        
        label, state_length = batch["label"].to(self.device), batch["state_length"].to(self.device)

        with torch.no_grad():
            next_state_max_actions_model = torch.argmax(self.predictor(next_state)[0],dim = -1,keepdim= True)
            next_state_values_lag_model = self.lag_predictor(next_state)[0]
            next_state_values_for_max_action = torch.gather(input= next_state_values_lag_model, dim= 1, index= next_state_max_actions_model) # (BS,1)
            next_state_values_for_max_action = next_state_values_for_max_action*(~(is_terminal.unsqueeze(-1)))
            target = reward + lam*(next_state_values_for_max_action.squeeze()) # (BS)
        
        predicted_values = self.predictor(state)[0]
        predicted_values_for_taken_action = torch.gather(input= predicted_values, dim= 1,index= action.unsqueeze(-1)).squeeze() # (BS)
        
        q_loss = self.mse_loss_function(target, predicted_values_for_taken_action).mean()


        # adding hint loss
        hint_loss = self.hintLoss(model_out= predicted_values, labels= label, state_lengths= state_length)

        loss = q_loss*self.q_loss_alpha + self.hint_loss_alpha*hint_loss
        loss.backward()
        predictor_optimizer.step()
        predictor_optimizer.zero_grad()
        self.logger.addMetric(metric_name= "q_loss", value= loss.item())
        self.logger.addMetric(metric_name= "hint_loss", value= hint_loss.item())


        if steps%self.model_replacement_steps == 0:
            self.__refreshLagModel()
                
        return loss
        
    def __refreshLagModel(self):
        self.lag_predictor = deepcopy(self.predictor)
        self.lag_predictor.eval()
    


    def eval(self,dataset : BaseFlowDataset):
        metrices = self.evaluator.getMetrices(dataset= dataset,ood_dataset= None)
        return metrices["macro_f1"],metrices["time"],metrices["incorrect_ood"]
    

    def evalTrain(self):
        f1,average_time,incorrect_ood = self.eval(dataset= self.train_dataset)
        self.logger.addMetric(metric_name= "train_eval_f1", value= f1)
        self.logger.addMetric(metric_name= "train_eval_time", value= average_time)
        self.logger.addMetric(metric_name= "incorrect_ood_train", value = incorrect_ood)

    def evalTest(self):
        f1,average_time,incorrect_ood = self.eval(dataset= self.test_dataset)

        if f1 >= self.best["score"]:
            self.best["score"] = f1
            self.best["model"] = deepcopy(self.predictor)
        
        self.logger.addMetric(metric_name= "test_eval_f1", value= f1)
        self.logger.addMetric(metric_name= "test_eval_time", value= average_time)
        self.logger.addMetric(metric_name= "incorrect_ood_test", value= incorrect_ood)

    def evalOOD(self):
        metrices = self.evaluator.getMetrices(ood_dataset= self.ood_dataset, dataset= None)
        self.logger.addMetric(metric_name= "ood_eval", value= metrices["ood_accuracy"])
        self.logger.addMetric(metric_name= "ood_eval_time", value= metrices["ood_time"])
        

        

    def train(self,epochs : int,batch_size = 64,lr = .001,lam = .99):
        # TODO add batch_sampler
        """
        Can stress enough how important the shuffle == True is in the Dataloader
        """
        if self.use_sampler == False:
            train_dataloader = DataLoader(dataset= self.memory_dataset,collate_fn= self.memory_dataset.collateFn,batch_size= batch_size,drop_last= True,shuffle= True)
        else:
            sampler = self.getWeightedSampler(memory_dataset= self.memory_dataset)
            train_dataloader = DataLoader(dataset= self.memory_dataset, collate_fn= self.memory_dataset.collateFn, batch_size= batch_size,drop_last= True,sampler= sampler)
        predictor_optimizer = torch.optim.Adam(params= self.predictor.parameters(), lr = lr)
        steps = 0

        for epoch in range(epochs):
            for batch in train_dataloader:
                self.trainStep(steps = steps,batch= batch,lam= lam, predictor_optimizer= predictor_optimizer)
                steps += 1            

                if steps%1000 == 0:
                    self.evalTest()
                    if self.ood_dataset != None:
                        self.evalOOD()
                if steps%2000 == 0:
                    self.evalTrain()
                


