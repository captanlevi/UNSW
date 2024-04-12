from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def evaluateModelOnDataSet(dataset ,model : nn.Module,device : str,calc_f1 = True):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        model.eval()
        with torch.no_grad():
            labels = []
            predictions = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(device), batch["label"].to(device)
                batch_predicted = torch.argmax(model(batch_X),dim= -1).cpu().numpy().tolist() # (BS,seq_len)
                predictions.extend(batch_predicted)
                labels.extend(batch_y.cpu().numpy().tolist())


        model.train()
        if calc_f1:
            _,_,f1,_ = precision_recall_fscore_support(labels, predictions, average= "weighted",zero_division=0)
            return f1
        else:
            return np.array(predictions)


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