from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def evaluateModelOnDataSet(dataset ,model : nn.Module,device : str,calc_f1 = True):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        model.eval()
        with torch.no_grad():
            labels = []
            predictions = []
            last_pred_scores = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(device), batch["label"].to(device)
                model_out_probs = F.softmax(model(batch_X)[-1],dim= -1)
                batch_predicted = torch.argmax(model_out_probs,dim= -1).cpu().numpy().tolist() # (BS,seq_len)
                batch_last_pred_scores = model_out_probs[:,-1].cpu().numpy().tolist()
                predictions.extend(batch_predicted)
                last_pred_scores.extend(batch_last_pred_scores)
                labels.extend(batch_y.cpu().numpy().tolist())


        model.train()
        last_pred_scores = np.array(last_pred_scores)
        if calc_f1:
            _,_,f1,_ = precision_recall_fscore_support(labels, predictions, average= "weighted",zero_division=0)
            return f1,last_pred_scores.mean()
        else:
            return np.array(predictions),last_pred_scores.mean()


def evaluateModelOnDataSetFeature(dataset ,feature_extractor : nn.Module,classifier : nn.Module,device : str):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        feature_extractor.eval()
        classifier.eval()
        with torch.no_grad():
            labels = []
            predictions = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(device), batch["label"].to(device)
                batch_predicted = torch.argmax(classifier(feature_extractor(batch_X)),dim= -1).cpu().numpy().tolist() # (BS,seq_len)
                predictions.extend(batch_predicted)
                labels.extend(batch_y.cpu().numpy().tolist())


        classifier.train()
        feature_extractor.train()
        _,_,f1,_ = precision_recall_fscore_support(labels, predictions, average= "weighted",zero_division=0)
        return f1





class Evaluator:
    def __init__(self,model,device):
        self.model = model.to(device)
        self.device = device


    
    def classificationScores(self,predicted,y_true,labels):
        predicted,y_true,labels = np.array(predicted),np.array(y_true), np.array(labels)
        _,_,micro_f1,_ = precision_recall_fscore_support(y_true = y_true, y_pred= predicted, average= "micro",zero_division=0)
        _,_,macro_f1,_ = precision_recall_fscore_support(y_true= y_true, y_pred= predicted, average= "macro",zero_division=0)
        incorrect_ood = (predicted == -1).sum()/ len(predicted)
        accuracy = accuracy_score(y_true= y_true, y_pred= predicted)
        cm = confusion_matrix(y_true= y_true, y_pred= predicted, labels= labels)
        return dict(micro_f1 = micro_f1,macro_f1 = macro_f1, accuracy = accuracy, cm = cm, incorrect_ood = incorrect_ood)
    

    def predictOnDataset(self,dataset):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        self.model.eval()
        with torch.no_grad():

            labels = []
            predictions = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(self.device), batch["label"].to(self.device)
                predicted = self.model(batch_X)[0] # (BS,num_class)
                predicted = torch.argmax(predicted,dim= -1).cpu().numpy()
                
                predictions.extend(predicted)
                labels.extend(batch_y.cpu().numpy().tolist())


        self.model.train()
        return predictions,labels
    

    def getMetrices(self,dataset):
        predictions,y_true = self.predictOnDataset(dataset= dataset)
        labels = list(range(0,len(dataset.label_to_index)))
        metrices =  self.classificationScores(predicted= predictions,y_true= y_true,labels= labels)
        return metrices

        


class EarlyEvaluation(Evaluator):
    def __init__(self,min_steps,device,model):
        super().__init__(model= model,device= device)
        self.min_steps = min_steps
    

    def __processSinglePrediction(self,prediction,num_classes):
        """
        predictions are of shape (seq_len)
        """
        
        for time in range(self.min_steps,len(prediction)):
            if prediction[time] < num_classes:
                return (prediction[time],time + 1)
        
        return (-1,len(prediction))



    def predictOnDataset(self,dataset):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        self.model.eval()
        with torch.no_grad():

            labels = []
            predictions_time = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(self.device), batch["label"].to(self.device)
                predicted = self.predictStep(batch_X = batch_X).cpu().numpy() # (BS,seq_len)
                processed_predictions = map(lambda x : self.__processSinglePrediction(x,len(dataset.label_to_index)), predicted)

                predictions_time.extend(processed_predictions)
                labels.extend(batch_y.cpu().numpy().tolist())


        self.model.train()
        predictions_time = np.array(predictions_time)
        predictions, time = predictions_time[:,0], predictions_time[:,1]

        return predictions,time,labels
    

    def predictStep(self,batch_X):
        """
        Here the return is of shape (BS,seq_len)
        """
        self.model.eval()
        with torch.no_grad():
            model_out = self.model.earlyClassificationForward(batch_X)[0]
        self.model.train()
        return torch.argmax(model_out,dim= -1)
    



    def getMetrices(self, dataset, ood_dataset = None):
        metrices = dict()

        if dataset != None:
            predictions,time,y_true = self.predictOnDataset(dataset= dataset)
            labels = list(range(0,len(dataset.label_to_index)))
            metrices = self.classificationScores(predicted= predictions, labels= labels, y_true= y_true)
            metrices["time"] =  time.mean()

        if ood_dataset != None:
            predictions,time,_ = self.predictOnDataset(dataset= ood_dataset)
            
            ood_accuracy = (predictions == -1).sum()/len(predictions)
            ood_time = time.mean()

            metrices["ood_accuracy"] = ood_accuracy
            metrices["ood_time"] = ood_time

        return metrices


        