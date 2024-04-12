from ...modeling.neuralNetworks import TransformerGenerator,CNNNetwork, LSTMNetwork, TransformerFeatureGenerator
from .datasets import SineWaveDataset
from ...modeling.loggers import Logger
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from .utils import BatchReplacer
from ...utils.evaluations import evaluateModelOnDataSetFeature,evaluateModelOnDataSet
import numpy as np
import random
from ...utils.commons import augmentData
from copy import deepcopy

class GANTrainer:
    def __init__(self,generator : TransformerGenerator,discriminator : CNNNetwork,logger : Logger,device = None,grad_clip = 10):
        self.device = device
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.logger = logger
        self.grad_clip = grad_clip

    
    def trainDiscriminator(self,discriminator,real_batch,fake_batch,discriminator_optimizer,gp_weight : float,layer: int):
        # fake batch is already detached
        discriminator.train()
        discriminator_optimizer.zero_grad()
        discriminator_loss,cost_wd,grad_penalty = self.calcDiscriminatorLoss(discriminator= discriminator,real_data= real_batch,generated_data= fake_batch,gp_weight= gp_weight)
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.grad_clip)
        discriminator_optimizer.step()
        self.logger.addMetric(metric_name= "{}_discriminator_loss".format(layer), value= discriminator_loss.cpu().item())
        self.logger.addMetric(metric_name= "{}_cost_wd".format(layer), value= cost_wd.cpu().item())
        self.logger.addMetric(metric_name= "{}_gp_loss".format(layer), value= grad_penalty.cpu().item())
        
    def trainGenerator(self,real_batch,generator_optimizer):
        self.generator.train()
        batch_size= real_batch.shape[0]
        Z = self.generator.generateRandomZ(batch_size = batch_size).to(self.device)
        fake_batch = self.generator(Z)
        fake_batch_model_out = self.discriminator(fake_batch)
        generator_optimizer.zero_grad()
        generator_loss = -(fake_batch_model_out.mean())
        generator_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
        generator_optimizer.step()
        self.logger.addMetric(metric_name= "generator_loss", value= generator_loss.cpu().item())


    def trainStep(self,step,n_critic,real_batch,generator_optimizer,discriminator_optimizer,gp_weight):
        with torch.no_grad():
            fake_batch = self.generator(self.generator.generateRandomZ(batch_size= real_batch.shape[0]).to(self.device)).detach()
        self.trainDiscriminator(real_batch= real_batch,discriminator_optimizer= discriminator_optimizer,
                                gp_weight= gp_weight,fake_batch= fake_batch,layer= 0,discriminator= self.discriminator)
        if step%n_critic == 0:
            self.generator.train()
            self.trainGenerator(real_batch= real_batch, generator_optimizer=generator_optimizer)
      


    def calcDiscriminatorLoss(self,discriminator,real_data, generated_data,gp_weight):
        assert real_data.shape == generated_data.shape
        batch_size = real_data.shape[0]
        disc_out_gen = discriminator(generated_data)
        disc_out_real = discriminator(real_data)

        alpha_shape = [batch_size]
        for i in range(len(real_data.shape) - 1):
            alpha_shape.append(1)

        alpha = torch.rand(alpha_shape).to(self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = (1 - alpha) * real_data.data + alpha * generated_data.data
        interpolated = interpolated.requires_grad_(True)

        # calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated)
        ones = torch.ones(prob_interpolated.size()).to(self.device)
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True)[0]
        
        # calculate gradient penalty
        grad_penalty = (
            torch.mean((gradients.contiguous().view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2)
        )


        cost_wd = disc_out_gen.mean() - disc_out_real.mean()
        cost = cost_wd + grad_penalty*gp_weight

        return cost, cost_wd,grad_penalty

    def plotGeneratedSample(self,z):
        self.generator.eval()
        with torch.no_grad():
            generated = self.generator(z)
            score_id = self.discriminator(generated)
            print(score_id)
            plt.imshow(generated[0][0].cpu().numpy())
            plt.show()
        self.generator.train()



    def train(self,train_dataset : SineWaveDataset,epochs,batch_size,n_critic,gp_weight = .1,lr_generator = 1e-5, lr_discriminator = 1e-4):
        train_dataloader = DataLoader(dataset= train_dataset,batch_size= batch_size,drop_last= True,shuffle= True)
        generator_optimizer = torch.optim.RMSprop(params= self.generator.parameters(), lr= lr_generator)
        discriminator_optimizer = torch.optim.RMSprop(params= self.discriminator.parameters(), lr= lr_discriminator)
        
        testing_sample = self.generator.generateRandomZ(batch_size= 1).to(self.device)

        step = 0
        for epoch in range(epochs):
            for batch in train_dataloader:

                step += 1
                self.trainStep(step= step,n_critic= n_critic,real_batch= batch["data"].float().to(self.device),generator_optimizer= generator_optimizer,
                                discriminator_optimizer= discriminator_optimizer,gp_weight= gp_weight)
            

                if step%2500 == 0:
                    self.plotGeneratedSample(testing_sample)









class OODTrainer(GANTrainer):
    def __init__(self,classifier : LSTMNetwork,generator: TransformerGenerator, discriminator: CNNNetwork, logger: Logger, device : str,n : int = 2,classifier_only = False):
        super().__init__(generator= generator, discriminator= discriminator, logger= logger, device= device)
        self.classifier = classifier.to(self.device)                                    
        self.kl_loss = nn.KLDivLoss(reduction="batchmean",log_target= True)              
        self.logger.setMetricReportSteps(metric_name= "train_f1", step_size= 1)          
        self.logger.setMetricReportSteps(metric_name= "test_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_accuracy", step_size= 1)           
        self.classifier_only = classifier_only 
        self.mse_loss = nn.MSELoss()
        self.discriminators = [discriminator.to(device)]
        self.generators = [generator.to(device)]

        self.n = n
        for _ in range(n-1):
            self.generators.append(deepcopy(generator).to(device))
            self.discriminators.append(deepcopy(discriminator).to(device))

        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def plotGeneratedSample(self,z):

        with torch.no_grad():
            for i in range(self.n):
                self.generators[i].eval()
                generated = self.generators[i](z)
                score_id = self.discriminators[i](generated)
                self.generators[i].train()
                print(score_id)
                #plt.imshow(generated[0,0,:,:].cpu().numpy())
                plt.plot(generated[0][:,0].cpu().numpy())
                plt.show()
       

    def trainClassifier(self,batch,labels,classifier_optimizer):
        self.classifier.train()
        classifier_optimizer.zero_grad()
        model_out = self.classifier(batch)
        loss = self.cross_entropy_loss(model_out,labels)

        numpy_batch = batch.cpu().numpy()
        np.random.shuffle(numpy_batch)
        augmented_batch = torch.tensor(numpy_batch).float().to(self.device)
        augmented_labels = torch.ones_like(labels)*(model_out.shape[-1] -1)
        augmented_labels = augmented_labels.to(self.device)
       

        augmented_model_out = self.classifier(augmented_batch)
        augmented_loss = self.cross_entropy_loss(augmented_model_out,augmented_labels)
        loss.backward()
        augmented_loss.backward()
        classifier_optimizer.step()
        self.logger.addMetric(metric_name= "classifier_loss", value= loss.cpu().item())
        self.logger.addMetric(metric_name= "classifier_aug_loss", value= augmented_loss.cpu().item())


    
    def _ganTraining(self,real_batch,step,n_critic,classifier_optimizer,discriminator_optimizers,generator_optimizers,gp_weight):
        
        def sampleBatch(pseudo_real_batch,batch_size):
            indices = np.arange(pseudo_real_batch.shape[0])
            chosen_indices = np.random.choice(a = indices,size= batch_size,replace= False)
            chosen_indices = torch.tensor(chosen_indices).to(self.device)
            return pseudo_real_batch[chosen_indices]

        batch_size = real_batch.shape[0]
        pseudo_real_batch = real_batch
        for i in range(self.n):
            with torch.no_grad():
                pseudo_fake_batch = self.generators[i](self.generators[i].generateRandomZ(batch_size=batch_size).to(self.device)).detach()
            feed_real_batch = sampleBatch(pseudo_real_batch,batch_size= batch_size)
            self.trainDiscriminator(discriminator= self.discriminators[i],real_batch= feed_real_batch,
                                    fake_batch= pseudo_fake_batch,layer= i,
                                    discriminator_optimizer= discriminator_optimizers[i],gp_weight= gp_weight)

            pseudo_real_batch = torch.concatenate([pseudo_real_batch,pseudo_fake_batch], dim= 0).to(self.device)
            #pseudo_real_batch = pseudo_fake_batch

        if step%n_critic == 0:
            Z = self.generators[0].generateRandomZ(batch_size= batch_size).to(self.device)
            for i in range(self.n):
                self.trainGenerator(generator= self.generators[i],discriminator= self.discriminators[i],Z = Z,
                                    generator_optimizer=generator_optimizers[i],classifier_optimizer= classifier_optimizer,layer= i)

    def trainStep(self, step, n_critic, real_batch,classifier_optimizer,generator_optimizers, discriminator_optimizers,gp_weight):

        real_batch_X,real_batch_y = real_batch["data"].float().to(self.device), real_batch["label"].to(self.device)
        self.trainClassifier(batch= real_batch_X,labels = real_batch_y,classifier_optimizer= classifier_optimizer)
        if self.classifier_only:
            return
        self._ganTraining(real_batch= real_batch_X,step= step,n_critic= n_critic,generator_optimizers=generator_optimizers,
                           discriminator_optimizers=discriminator_optimizers,classifier_optimizer= classifier_optimizer,gp_weight= gp_weight)
        
      



    def trainGenerator(self,generator,discriminator,Z,generator_optimizer,classifier_optimizer,layer : int):
        generator.train()
        self.classifier.train()
        generator_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        fake_batch = generator(Z)
        fake_batch_model_out = discriminator(fake_batch)
        gan_generator_loss = -(fake_batch_model_out.mean()) # as close to real dist as possible
        
        classifier_out = self.classifier(fake_batch) # taking the prob of unknown class
        classifier_labels = torch.ones(fake_batch.shape[0])*(classifier_out.shape[-1] -1) # I want to prob to be in the last unknown label
        classifier_labels = classifier_labels.long().to(self.device)
        log_generator_loss = self.cross_entropy_loss(classifier_out,classifier_labels)   

        generator_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        generator_loss = log_generator_loss + gan_generator_loss
        generator_loss.backward()
        generator_optimizer.step()
        #if layer == self.n -1:
        # only train classifier on the last_layer
        classifier_optimizer.step()
        self.logger.addMetric(metric_name= "{}_generator_log_loss".format(layer), value= log_generator_loss.cpu().item())
        self.logger.addMetric(metric_name= "{}_generator_loss".format(layer), value= generator_loss.cpu().item())

    def train(self,train_dataset,test_dataset,ood_dataset,epochs,batch_size,n_critic,gp_weight = .1,lr_classifier = 1e-4,lr_generator = 1e-5, lr_discriminator = 1e-4):
        train_dataloader = DataLoader(dataset= train_dataset,batch_size= batch_size,drop_last= True,shuffle= True)

        discriminator_optimizers = list(map(lambda x : torch.optim.RMSprop(params= x.parameters(), lr= lr_discriminator),self.discriminators))
        generator_optimizers = list(map(lambda x : torch.optim.RMSprop(params= x.parameters(), lr= lr_generator),self.generators))

        classifier_optimizer = torch.optim.Adam(params= self.classifier.parameters(),lr= lr_classifier)
        testing_sample = self.generators[0].generateRandomZ(batch_size= 1).to(self.device)

        step = 0
        for epoch in range(epochs):
            for batch in train_dataloader:
                step += 1

                self.trainStep(step= step,n_critic= n_critic,real_batch= batch,generator_optimizers= generator_optimizers,classifier_optimizer= classifier_optimizer,
                                discriminator_optimizers= discriminator_optimizers,gp_weight= gp_weight)
            

                if step%1500 == 0:
                    train_f1 = evaluateModelOnDataSet(dataset= train_dataset,model= self.classifier,device= self.device)
                    self.logger.addMetric(metric_name= "train_f1", value= train_f1)
                    test_f1 = evaluateModelOnDataSet(dataset= test_dataset,model= self.classifier,device= self.device)
                    self.logger.addMetric(metric_name= "test_f1", value= test_f1)
                    if ood_dataset != None:
                        predictions = evaluateModelOnDataSet(dataset= ood_dataset,model= self.classifier,device= self.device,calc_f1= False)
                        ood_accuracy = (predictions == len(train_dataset.label_to_index)).sum()/ len(predictions)
                        self.logger.addMetric(metric_name= "ood_accuracy", value= ood_accuracy)
                    
                    self.plotGeneratedSample(testing_sample)
   


