import torch.nn as nn
from .loggers import Logger
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from ..flowUtils.flowDatasets import BaseFlowDataset
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
import os
from joblib import dump as joblib_dump, load as joblib_load

class BaseModel:
    def __init__(self,model,train_dataset : BaseFlowDataset,test_dataset : BaseFlowDataset,model_dir_path : str,model_name : str):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_dir_path = model_dir_path
        self.model_name = model_name
    
    def predictOnDataset(self,dataset: BaseFlowDataset):
        raise NotImplementedError()

    def calcF1(self,dataset : BaseFlowDataset):
        labels = list(map(lambda x : x["label"], dataset))
        preds = self.predictOnDataset(dataset)
        _,_,f1,_ = precision_recall_fscore_support(labels, preds, average= "macro",zero_division=0)
        return f1
    
    def calcTestF1(self):
        return self.calcF1(dataset= self.test_dataset)

    def calcTrainF1(self):
        return self.calcF1(dataset= self.train_dataset)
    

    def plotConfusionMatrix(self,dataset : BaseFlowDataset):
        display_labels = []
        for key,value in dataset.label_to_index.items():
            display_labels.append([value,key])
        display_labels.sort()
        display_labels = list(map(lambda x : x[1],display_labels))
        labels = list(map(lambda x : x["label"], dataset))
        preds = self.predictOnDataset(dataset)

      
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels= display_labels)
        disp.plot()
        plt.show()

    def getModelPath(self):
        return os.path.join(self.model_dir_path,self.model_name)

    def saveModel(self):
        raise NotImplementedError()

    def loadModel(self):
        raise NotImplementedError()





class RandomForestModel(BaseModel):
    def __init__(self, model : RandomForestClassifier, train_dataset: BaseFlowDataset, test_dataset: BaseFlowDataset,model_dir_path : str, model_name : str):
        super().__init__(model, train_dataset, test_dataset,model_dir_path= model_dir_path, model_name= model_name)
    

    def train(self):
        X_train = list(map(lambda x : x["flow"], self.train_dataset)), 
        y_train = list(map(lambda x : x["label"], self.train_dataset))
        self.model.fit(X_train,y_train)

    
    def predictOnDataset(self, dataset: BaseFlowDataset):
        X_test = list(map(lambda x : x["flow"], dataset))
        return self.model.predict(X_test)


    def saveModel(self):
        model_path = self.getModelPath()
        joblib_dump(self.model,model_path)
    def loadModel(self):
        model_path = self.getModelPath()
        self.model = joblib_load(model_path)




class LSTMModel(BaseModel):
    def __init__(self,model : nn.Module,train_dataset : BaseFlowDataset,test_dataset : BaseFlowDataset,logger : Logger,model_dir_path : str, model_name : str):
        super().__init__(model= model, train_dataset= train_dataset,test_dataset= test_dataset, model_dir_path= model_dir_path, model_name= model_name)
        self.logger = logger


    def trainStep(self,batch_X,batch_y):
        model_out = self.model(batch_X)
        loss = self.model.loss_function(model_out,batch_y)
        return loss

    def predictStep(self,batch_X):
        model_out = self.model(batch_X)
        return torch.argmax(model_out,dim= -1)

    def train(self,epochs,batch_size,lr):
        optimizer = torch.optim.Adam(params= self.model.parameters(), lr= lr)
        train_dataloader = DataLoader(dataset= self.train_dataset,shuffle= True,drop_last= True,batch_size= batch_size)
        for epoch in range(epochs):
            epoch_loss = 0
            counter = 0
            for batch in train_dataloader:
                batch_X,batch_y = batch["flow"].float(), batch["label"]
                loss = self.trainStep(batch_X= batch_X, batch_y= batch_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                counter += 1
            epoch_loss /= counter

            #print("epoch {} loss = {}".format(epoch,epoch_loss))
            self.logger.addMetric(metric_name= "train_loss", value= epoch_loss)
            if epoch%10 == 0:
                test_f1  =  self.calcTestF1()
                self.logger.addMetric("test_f1", test_f1)
            if epoch%20 == 0:
                train_f1 = self.calcTrainF1()
                self.logger.addMetric("train_f1", train_f1)
            

    def predictOnDataset(self,dataset):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        with torch.no_grad():
            predictions = []
            labels = []

            for batch in dataloader:
                batch_X,batch_y = batch["flow"].float(), batch["label"]
                predicted = self.predictStep(batch_X = batch_X)
                predictions.extend(predicted.numpy().tolist())
                labels.extend(batch_y.numpy().tolist())
        return predictions
        


    def saveModel(self):
        model_path = self.getModelPath()
        torch.save(self.model.cpu().state_dict(), model_path + ".pth")
    
    def loadModel(self):
        model_path = self.getModelPath()
        self.model = torch.load(model_path + ".pth")