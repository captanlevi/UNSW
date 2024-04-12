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

class BaseClassificationTrainer:
    def __init__(self,model,model_dir_path : str,model_name : str):
        self.model = model
        self.model_dir_path = model_dir_path
        self.model_name = model_name
    
    def predictOnDataset(self,dataset: BaseFlowDataset):
        raise NotImplementedError()

    def calcF1(self,dataset : BaseFlowDataset):
        labels = list(map(lambda x : x["label"], dataset))
        preds = self.predictOnDataset(dataset)
        _,_,f1,_ = precision_recall_fscore_support(labels, preds, average= "macro",zero_division=0)
        return f1
    

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





class RandomForestModel(BaseClassificationTrainer):
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



class NNClassificationTrainer(BaseClassificationTrainer):
    def __init__(self,feature_extractor,classifier,device,logger : Logger):
        self.feature_extractor = feature_extractor.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.logger = logger
        self.cross_entropy_loss =  nn.CrossEntropyLoss()

        self.logger.setMetricReportSteps(metric_name= "train_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "test_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_loss", step_size= 10)

    

    def trainStep(self,batch,feature_extractor_optimizer,classifier_optimizer):
        X,y = batch["data"].float().to(self.device), batch["label"].to(self.device)
        model_out = self.classifier(self.feature_extractor(X))
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        loss = self.cross_entropy_loss(model_out,y).mean()
        loss.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        self.logger.addMetric(metric_name= "train_loss", value= loss.cpu().item())
        return loss

    def predictStep(self,batch_X):
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            model_out = self.classifier(self.feature_extractor(batch_X.float().to(self.device)))
        self.feature_extractor.train()
        self.classifier.train()
        return torch.argmax(model_out,dim= -1)

    def train(self,train_dataset,test_dataset,epochs,batch_size,lr):
        feature_extractor_optimizer = torch.optim.Adam(params= self.feature_extractor.parameters(), lr= lr)
        classifier_optimizer =  torch.optim.Adam(params= self.classifier.parameters(), lr= lr)
        step = 0
        train_dataloader = DataLoader(dataset= train_dataset,shuffle= True,drop_last= True,batch_size= batch_size)
        for epoch in range(epochs):
            for batch in train_dataloader:
                self.trainStep(batch= batch,feature_extractor_optimizer=feature_extractor_optimizer,
                               classifier_optimizer= classifier_optimizer)
                
                if step%500 == 0:
                    test_f1  =  self.calcF1(test_dataset)
                    self.logger.addMetric("test_f1", test_f1)
                if step%1000 == 0:
                    train_f1 = self.calcF1(train_dataset)
                    self.logger.addMetric("train_f1", train_f1)
                step += 1

    def predictOnDataset(self,dataset):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        with torch.no_grad():
            predictions = []
            labels = []

            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(self.device), batch["label"].to(self.device)
                predicted = self.predictStep(batch_X = batch_X)
                predictions.extend(predicted.cpu().numpy().tolist())
                labels.extend(batch_y.cpu().numpy().tolist())
        return predictions
    

    def saveModel(self):
        model_path = self.getModelPath()
        torch.save(self.feature_extractor.cpu().state_dict(), model_path + ".pth")
    
    def loadModel(self):
        model_path = self.getModelPath()
        self.feature_extractor = torch.load(model_path + ".pth")