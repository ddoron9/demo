import queue
import matplotlib.pyplot as plt
from threading import Thread
import librosa
import numpy as np
import pickle
import os, sys

import torch
import torch.nn as nn
import time

print(os.path.abspath('.'))  

from models.audio_emotion import *
from models.audio_gender import *
from utils.audio_utils import *
from preprocess import AudioData

global emotion_classes 
emotion_classes = {0:'neutral', 1:'sad', 2:'surprise', 3:'happy', 4:'angry'} 
global gender_classes
gender_classes = ['Female','Male']
global intention_classes
intention_classes = {0 : '요청-위치안내', 1 : '요청-기타안내', 2 : '질문-위치', 3 : '질문-기타', 4 : '인사', 5 : '진술'}
global SAMPLING_RATE 
SAMPLING_RATE = 48000   


PATH = '/opt/utopsoft/label-management/webapps/attach/label'
transformed_train_data = AudioData(PATH,SAMPLING_RATE)

def audio_emotion_class(video): 
     

    gender_outputs = predict(gender_model,imagesTensor.to(device))
    emotion_outputs = predict(emotion_model,imagesTensor.to(device))
    
    prediction = torch.argmax(emotion_outputs[1],dim=1)
    gender = torch.argmax(gender_outputs,dim=1)
  


def train_model(model, criterion , optimizer, n_epochs=25):
  # from torch.utils.data import  TensorDataset, DataLoader
  #x y 따로 불러도 Tensordataset 사용하면 모을 수 있음
  # transformed_train_data == gender intention tone
    mel = torch.load('/Users/doyi/Downloads/mel-20-3-64-256.pt')
    embedded_text = torch.load('/Users/doyi/Downloads/text-20-5-200.pt')
    label = torch.load('/Users/doyi/Downloads/label-20-5-200.pt')
    print(f'loaded image size : {mel.size()} \n label size : {label.size()} \n embedded text : {embedded_text.size()}')
    transformed_train_data = TensorDataset(mel,embedded_text,label)
    train_dataloader = DataLoader(transformed_train_data, batch_size=50, shuffle=True, num_workers=4) 


    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(train_dataloader):
            # importing data and moving to GPU
            #image, text, label1, label2, label3
            data = sample_batched[0].to(device),\
                                            sample_batched[1].to(device),\
                                            sample_batched[2][0].to(device),\
                                            sample_batched[2][1].to(device) ,\
                                            sample_batched[2][2].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            output =model(data[0],data[1])

                   
            # calculate loss
            loss = 0.0
            for i in range(3):
                loss +=criterion(output[i], data[2+i].squeeze().type(torch.LongTensor)) 
             
            # back prop
            loss.backward()
            # grad
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 50 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
        # validate the model #
        model.eval()
        for batch_idx, sample_batched in enumerate(test_dataloader):
            data = sample_batched[0].to(device),\
                                            sample_batched[1].to(device),\
                                            sample_batched[2][0].to(device),\
                                            sample_batched[2][1].to(device) ,\
                                            sample_batched[2][2].to(device)
             
            output_logits, emotion_softmax , intention_softmax =model(data[0],data[1])
            loss = 0.0  
            # calculate loss
            for i in range(3):
              loss += criterion(output[i], data[2+i].squeeze().type(torch.LongTensor))
         
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model, 'model.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
    # return trained model
    return model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Selected device is {}'.format(device))



model = ParallelModel(num_emotions=6,num_intentions = 5).to(device)
print('Number of trainable params: ',sum(p.numel() for p in model.parameters()))
 
#For multilabel output: race and age
criterion_multioutput = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model =train_model(model, criterion_multioutput, optimizer)

SAVE_PATH = os.path.join(os.getcwd(),'models')
os.makedirs('models',exist_ok=True)
torch.save(model.state_dict(),os.path.join(SAVE_PATH,'cnn_transf_parallel_model.pt'))
print('Model is saved to {}'.format(os.path.join(SAVE_PATH,'cnn_transf_parallel_model.pt')))